"""Exercise target discovery and deployment ordering without cloud access."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "runs_xuesong/scripts/deploy_aime_7b_worker.sh"


class Aime7bDeployTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    self.repo = self.root / "Project_7B/tunix"
    self.repo.mkdir(parents=True)
    self.bin = self.root / "bin"
    self.bin.mkdir()
    self.calls_file = self.root / "cloud_calls.jsonl"
    self.nodes = self.root / "nodes.json"
    self.node = {
        "name": "projects/test-project/locations/us-test1-a/nodes/current-tpu",
        "networkEndpoints": [{"ipAddress": "10.0.0.1"}, {"ipAddress": "10.0.0.2"}],
    }
    self.nodes.write_text(json.dumps([self.node]))
    dependencies = self.root / "dependencies"
    dependencies.mkdir()
    for package in ("jax", "jaxlib", "libtpu", "vllm"):
      info = dependencies / f"{package}-0.0.dist-info"
      info.mkdir()
      (info / "METADATA").write_text(
          f"Metadata-Version: 2.1\nName: {package}\nVersion: 0.0\n"
      )
    self._executable(
        self.repo / ".venv/bin/python",
        "#!/bin/sh\nexport PYTHONPATH=" + shlex.quote(str(dependencies))
        + "\nexec " + shlex.quote(sys.executable) + ' "$@"\n',
    )
    self._executable(self.bin / "hostname", "#!/bin/sh\nprintf '10.0.0.1 127.0.0.1\\n'\n")
    self._executable(self.bin / "sha256sum", '#!/bin/sh\nexec shasum -a 256 "$@"\n')
    self._executable(
        self.bin / "curl",
        "#!" + sys.executable + "\n"
        "import os, sys\n"
        "if os.getenv('FAIL_METADATA'): sys.exit(7)\n"
        "url=sys.argv[-1]\n"
        "print('projects/123/zones/us-test1-a' if url.endswith('instance/zone') else 'test-project')\n",
    )
    self._executable(
        self.bin / "gcloud",
        "#!" + sys.executable + "\n"
        "import json, os, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['CLOUD_CALLS'], 'a') as out: out.write(json.dumps(args)+'\\n')\n"
        "if 'list' in args: print(pathlib.Path(os.environ['NODES_FILE']).read_text())\n"
        "elif 'scp' in args and os.getenv('FAIL_SCP'): sys.exit(23)\n",
    )
    script = self.repo / "runs_xuesong/scripts" / SCRIPT.name
    script.parent.mkdir(parents=True)
    script.write_bytes(SCRIPT.read_bytes())
    (self.repo / ".gitignore").write_text("/.venv/\n/runs_xuesong/cache/\n")
    (self.repo / "tracked.txt").write_text("committed source\n")
    self._git("init", "-q")
    self._git("config", "user.email", "check@example.invalid")
    self._git("config", "user.name", "Deployment check")
    self._git("add", ".")
    self._git("commit", "-qm", "fixture")
    self.env = dict(os.environ)
    for key in ("TPU_NAME", "ZONE", "TPU_PROJECT", "REMOTE_WORKER_INDEX", "FAIL_METADATA", "FAIL_SCP"):
      self.env.pop(key, None)
    self.env.update(
        REPO=str(self.repo),
        OLD_REPO=str(self.root / "Project/tunix"),
        PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
        CLOUD_CALLS=str(self.calls_file),
        NODES_FILE=str(self.nodes),
    )

  def _executable(self, path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)

  def _git(self, *args):
    return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

  def _run(self, *args, **env):
    return subprocess.run(
        ["bash", str(self.repo / "runs_xuesong/scripts" / SCRIPT.name), *args],
        env=dict(self.env, **env), text=True, capture_output=True,
    )

  def _calls(self):
    if not self.calls_file.exists():
      return []
    return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

  def _remote_bodies(self):
    bodies = []
    for call in self._calls():
      for arg in call:
        if arg.startswith("--command="):
          # printf %q can emit Bash ANSI-C quoting, which POSIX shlex cannot
          # decode. Intercept the bash invocation without executing its body.
          decoder = (
              'bash() { [[ "$1" == -lc && "$#" == 2 ]] || return 2; '
              'printf "%s" "$2"; }; '
              + arg.removeprefix("--command=")
          )
          bodies.append(subprocess.check_output(["bash", "-c", decoder], text=True))
    return bodies

  def test_missing_variables_are_discovered_without_remote_mutation(self):
    result = self._run("--dry-run")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("TPU=current-tpu worker=1 IP=10.0.0.2", result.stdout)
    self.assertIn("ZONE=us-test1-a", result.stdout)
    self.assertEqual(len(self._calls()), 1)
    self.assertIn("--project=test-project", self._calls()[0])
    self.assertIn("--zone=us-test1-a", self._calls()[0])
    self.assertNotIn("unbound variable", result.stderr)

  def test_wrong_explicit_node_is_rejected_before_ssh(self):
    result = self._run(TPU_NAME="old-tpu")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("found 0", result.stderr)
    self.assertEqual(len(self._calls()), 1)

  def test_ambiguous_nodes_are_rejected_before_ssh(self):
    other = dict(self.node, name="other-tpu")
    self.nodes.write_text(json.dumps([self.node, other]))
    result = self._run()
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("found 2", result.stderr)
    self.assertEqual(len(self._calls()), 1)

  def test_selecting_local_worker_is_rejected(self):
    result = self._run(REMOTE_WORKER_INDEX="0")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("Selected remote worker is this host", result.stderr)
    self.assertEqual(len(self._calls()), 1)

  def test_metadata_failure_explains_required_setting(self):
    result = self._run(FAIL_METADATA="1")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("Set ZONE", result.stderr)
    self.assertEqual(self._calls(), [])

  def test_dirty_checkout_stops_before_discovery(self):
    (self.repo / "tracked.txt").write_text("local changes\n")
    result = self._run()
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("local changes", result.stderr)
    self.assertEqual(self._calls(), [])

  def test_failed_copy_never_extracts_or_bootstraps(self):
    result = self._run(FAIL_SCP="1")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("file_transfer", result.stderr)
    bodies = self._remote_bodies()
    self.assertEqual(len(bodies), 1)
    self.assertNotIn("tar -xzf", bodies[0])
    self.assertNotIn("bootstrap_jason_env", bodies[0])

  def test_complete_order_and_generated_remote_commands(self):
    result = self._run()
    self.assertEqual(result.returncode, 0, result.stderr)
    kinds = [call[4] for call in self._calls()]
    self.assertEqual(kinds, ["list", "ssh", "scp", "scp", "scp", "ssh", "ssh"])
    bodies = self._remote_bodies()
    self.assertEqual(len(bodies), 3)
    self.assertIn("sha256sum -c archive.sha256", bodies[1])
    self.assertIn("sha256sum -c", bodies[1])
    self.assertIn(".deployed_git_head", bodies[1])
    self.assertIn("EXPECTED_VERSIONS", bodies[2])
    self.assertIn("deepseek_qwen_config_test.py", bodies[2])
    self.assertIn("Worker dependency versions differ", bodies[2])
    for body in bodies:
      subprocess.run(["bash", "-n"], input=body, text=True, check=True)
    sha = self._git("rev-parse", "HEAD").strip()
    deploy = self.repo / "runs_xuesong/cache/deploy" / sha
    check = subprocess.run(["shasum", "-a", "256", "-c", "archive.sha256"], cwd=deploy, capture_output=True)
    self.assertEqual(check.returncode, 0, check.stderr)

  def test_source_phase_does_not_create_environment(self):
    result = self._run("--phase", "source")
    self.assertEqual(result.returncode, 0, result.stderr)
    bodies = self._remote_bodies()
    self.assertEqual(len(bodies), 2)
    self.assertTrue(all("bootstrap_jason_env" not in body for body in bodies))


if __name__ == "__main__":
  unittest.main()
