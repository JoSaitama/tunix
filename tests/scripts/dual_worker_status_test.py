import pathlib
import subprocess
import textwrap
import unittest


class DualWorkerStatusTest(unittest.TestCase):

  def setUp(self):
    self.status_lib = (
        pathlib.Path(__file__).parents[2]
        / "runs_xuesong"
        / "scripts"
        / "dual_worker_status.sh"
    )

  def _run_bash(self, body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", textwrap.dedent(body)],
        check=False,
        capture_output=True,
        text=True,
        env={"STATUS_LIB": str(self.status_lib)},
    )

  def test_remote_program_success_overrides_ssh_transport_failure(self):
    result = self._run_bash(
        r"""
        set -euo pipefail
        source "$STATUS_LIB"
        status_log="$(mktemp)"
        trap 'rm -f "$status_log"' EXIT
        printf 'HOST=worker-1 STATUS=0\n' > "$status_log"
        (exit 1) &
        remote_pid=$!
        tunix_wait_and_capture_status ssh_status "$remote_pid"
        tunix_resolve_remote_status remote_status "$status_log" "$ssh_status"
        printf 'ssh=%s remote=%s\n' "$ssh_status" "$remote_status"
        test "$ssh_status" -eq 1
        test "$remote_status" -eq 0
        """
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout.strip(), "ssh=1 remote=0")

  def test_ssh_failure_is_used_when_program_status_is_missing(self):
    result = self._run_bash(
        r"""
        set -euo pipefail
        source "$STATUS_LIB"
        status_log="$(mktemp)"
        trap 'rm -f "$status_log"' EXIT
        (exit 7) &
        remote_pid=$!
        tunix_wait_and_capture_status ssh_status "$remote_pid"
        tunix_resolve_remote_status remote_status "$status_log" "$ssh_status"
        printf 'ssh=%s remote=%s\n' "$ssh_status" "$remote_status"
        test "$ssh_status" -eq 7
        test "$remote_status" -eq 7
        """
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout.strip(), "ssh=7 remote=7")


if __name__ == "__main__":
  unittest.main()
