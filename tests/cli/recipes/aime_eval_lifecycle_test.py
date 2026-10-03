"""CPU regression tests for evaluation cleanup and inherited output pipes."""

import argparse
import ast
from contextlib import ExitStack
import multiprocessing
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from unittest import mock

from examples.deepscaler import eval_sampler_lifecycle as lifecycle


ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT / "examples/deepscaler/eval_final_checkpoint_metrics.py"


class AimeEvalLifecycleTest(unittest.TestCase):

  def test_sampler_stops_on_success(self):
    sampler = mock.Mock()
    with lifecycle.managed_sampler(lambda: sampler) as result:
      self.assertIs(result, sampler)
      sampler.stop.assert_not_called()
    sampler.stop.assert_called_once_with()

  def test_generation_failure_still_stops_sampler(self):
    sampler = mock.Mock()
    with self.assertRaisesRegex(ValueError, "generation failed"):
      with lifecycle.managed_sampler(lambda: sampler):
        raise ValueError("generation failed")
    sampler.stop.assert_called_once_with()

  def test_shutdown_failure_is_not_reported_as_success(self):
    sampler = mock.Mock()
    sampler.stop.side_effect = RuntimeError("shutdown failed")
    with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
      with lifecycle.managed_sampler(lambda: sampler):
        pass

  def test_shutdown_timeout_is_bounded_and_reports_failure(self):
    release = threading.Event()
    sampler = mock.Mock()
    sampler.stop.side_effect = lambda: release.wait(5)
    started = time.monotonic()
    try:
      with self.assertRaises(TimeoutError):
        with lifecycle.managed_sampler(
            lambda: sampler, stop_timeout_seconds=0.01
        ):
          pass
      self.assertLess(time.monotonic() - started, 1)
    finally:
      release.set()

  @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork")
  def test_hard_exit_alone_leaves_inherited_pipe_open(self):
    body = textwrap.dedent("""
        import multiprocessing as mp
        import os
        import time
        ctx = mp.get_context("fork")
        ready = ctx.Event()
        def child():
          ready.set()
          time.sleep(0.5)
        worker = ctx.Process(target=child)
        worker.start()
        assert ready.wait(2)
        print("parent_exiting", flush=True)
        os._exit(0)
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", body], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
      self.assertEqual(process.wait(timeout=2), 0)
      with self.assertRaises(subprocess.TimeoutExpired):
        process.communicate(timeout=0.05)
    finally:
      process.communicate(timeout=3)

  @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork")
  def test_inherited_stdout_pipe_is_closed_before_hard_exit(self):
    # Real forked children inherit stdout, just like the TPU DP scheduler.
    # A separate, preexisting process must remain untouched by cleanup.
    body = textwrap.dedent("""
        import multiprocessing as mp
        import os
        import time
        from types import SimpleNamespace
        from examples.deepscaler.eval_sampler_lifecycle import managed_sampler
        ctx = mp.get_context("fork")
        unrelated = ctx.Process(target=time.sleep, args=(5,))
        unrelated.start()
        children = []
        def factory():
          for _ in range(2):
            child = ctx.Process(target=time.sleep, args=(5,))
            child.start()
            children.append(child)
          return SimpleNamespace(stop=lambda: None)
        with managed_sampler(factory, join_timeout_seconds=0.01):
          print("samples_done", flush=True)
        assert all(not child.is_alive() for child in children)
        assert unrelated.is_alive()
        unrelated.terminate()
        unrelated.join(1)
        print("clean_hard_exit", flush=True)
        os._exit(0)
    """)
    result = subprocess.run(
        [sys.executable, "-c", body], cwd=ROOT,
        capture_output=True, text=True, timeout=3, check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("remaining=0", result.stdout)
    self.assertIn("clean_hard_exit", result.stdout)

  @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork")
  def test_partial_factory_failure_reclaims_children(self):
    ctx = multiprocessing.get_context("fork")
    children = []
    def factory():
      child = ctx.Process(target=time.sleep, args=(5,))
      child.start()
      children.append(child)
      raise ValueError("factory failed")
    try:
      with self.assertRaisesRegex(ValueError, "factory failed"):
        with lifecycle.managed_sampler(factory, join_timeout_seconds=0.01):
          self.fail("Factory must fail before yield")
      self.assertFalse(children[0].is_alive())
    finally:
      for child in children:
        if child.is_alive():
          child.kill()
        child.join(1)

  def _eval_namespace(self, process_index=0):
    # Execute the actual orchestration functions without importing JAX/TPU.
    tree = ast.parse(EVALUATOR.read_text())
    names = {"run_eval", "_sync_eval_completion", "main"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    jax = mock.Mock()
    jax.process_index.return_value = process_index
    jax.process_count.return_value = 2
    namespace = {
        "argparse": argparse, "Any": object, "ExitStack": ExitStack,
        "jax": jax, "multihost_utils": mock.Mock(), "os": mock.Mock(),
        "sys": mock.Mock(), "_maybe_initialize_jax_distributed": mock.Mock(),
        "traceback": mock.Mock(),
        "_run_primary_eval": mock.Mock(return_value={"status": "done"}),
        "parse_args": mock.Mock(return_value=argparse.Namespace(disable_hard_exit=False)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVALUATOR), "exec"), namespace)
    return namespace

  def test_primary_cleanup_precedes_single_completion_barrier(self):
    ns = self._eval_namespace()
    events = []
    class Resource:
      def __enter__(self):
        return self
      def __exit__(self, *_):
        events.append("cleanup")
    def primary(args, resources):
      resources.enter_context(Resource())
      events.append("evaluation")
      return {"status": "done"}
    ns["_run_primary_eval"].side_effect = primary
    ns["multihost_utils"].sync_global_devices.side_effect = lambda _: events.append("barrier")
    result = ns["run_eval"](argparse.Namespace())
    self.assertEqual(result, {"status": "done"})
    self.assertEqual(events, ["evaluation", "cleanup", "barrier"])

  def test_primary_failure_releases_secondary_once(self):
    ns = self._eval_namespace()
    ns["_run_primary_eval"].side_effect = ValueError("evaluation failed")
    ns["main"]()
    ns["multihost_utils"].sync_global_devices.assert_called_once()
    ns["os"]._exit.assert_called_once_with(1)
    ns["traceback"].print_exc.assert_called_once()

  def test_cleanup_failure_releases_secondary_and_exits_nonzero(self):
    ns = self._eval_namespace()
    def primary(args, resources):
      resources.callback(mock.Mock(side_effect=RuntimeError("cleanup failed")))
      return {"status": "done"}
    ns["_run_primary_eval"].side_effect = primary
    ns["main"]()
    ns["multihost_utils"].sync_global_devices.assert_called_once()
    ns["os"]._exit.assert_called_once_with(1)

  def test_failed_barrier_is_not_retried(self):
    ns = self._eval_namespace()
    ns["multihost_utils"].sync_global_devices.side_effect = RuntimeError("barrier failed")
    ns["main"]()
    ns["multihost_utils"].sync_global_devices.assert_called_once()
    ns["os"]._exit.assert_called_once_with(1)

  def test_debug_mode_preserves_exception(self):
    ns = self._eval_namespace()
    ns["parse_args"].return_value.disable_hard_exit = True
    ns["_run_primary_eval"].side_effect = ValueError("evaluation failed")
    with self.assertRaisesRegex(ValueError, "evaluation failed"):
      ns["main"]()
    ns["multihost_utils"].sync_global_devices.assert_called_once()
    ns["os"]._exit.assert_not_called()

  def test_success_hard_exits_only_after_cleanup_and_barrier(self):
    ns = self._eval_namespace()
    ns["main"]()
    ns["multihost_utils"].sync_global_devices.assert_called_once()
    ns["os"]._exit.assert_called_once_with(0)

  def test_secondary_waits_without_constructing_sampler(self):
    ns = self._eval_namespace(process_index=1)
    result = ns["run_eval"](argparse.Namespace())
    self.assertEqual(result["status"], "secondary_eval_done")
    ns["_run_primary_eval"].assert_not_called()
    ns["multihost_utils"].sync_global_devices.assert_called_once()


if __name__ == "__main__":
  unittest.main()
