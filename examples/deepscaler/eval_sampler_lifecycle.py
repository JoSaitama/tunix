"""Bounded cleanup of a standalone evaluator's sampler and owned children."""

from contextlib import contextmanager
import multiprocessing
import os
import threading


def _phase(name, **fields):
  details = " ".join(f"{key}={value}" for key, value in fields.items())
  print(f"EVAL_TEARDOWN phase={name} pid={os.getpid()} {details}", flush=True)


def _reap_children(children, join_timeout_seconds):
  # Only processes created by this sampler factory are eligible. Never scan
  # machine-wide PIDs or signal another evaluator/training run.
  for child in children:
    child.join(timeout=join_timeout_seconds)
  for child in children:
    if child.is_alive():
      _phase("child_terminate", child_pid=child.pid, child_name=child.name)
      child.terminate()
  for child in children:
    child.join(timeout=join_timeout_seconds)
  for child in children:
    if child.is_alive():
      _phase("child_kill", child_pid=child.pid, child_name=child.name)
      child.kill()
  for child in children:
    child.join(timeout=join_timeout_seconds)
  remaining = [child.pid for child in children if child.is_alive()]
  _phase("children_reaped", remaining=len(remaining))
  if remaining:
    raise RuntimeError(f"Sampler children did not exit: {remaining}")


def _stop_sampler(sampler, children, stop_timeout_seconds, join_timeout_seconds):
  error = []
  stop = getattr(sampler, "stop", None)
  _phase("sampler_stop_begin", children=len(children))
  if callable(stop):
    def call_stop():
      try:
        stop()
      except BaseException as exc:  # Propagate after children are reclaimed.
        error.append(exc)

    # A stalled third-party shutdown must not prevent reclaiming its workers.
    thread = threading.Thread(target=call_stop, daemon=True)
    thread.start()
    thread.join(timeout=stop_timeout_seconds)
    if thread.is_alive():
      error.append(TimeoutError("Timed out stopping the evaluation sampler."))
  _reap_children(children, join_timeout_seconds)
  if error:
    _phase("sampler_stop_failed", error=type(error[0]).__name__)
    raise error[0]
  _phase("sampler_stop_complete")


@contextmanager
def managed_sampler(
    factory, *, stop_timeout_seconds=60.0, join_timeout_seconds=5.0
):
  """Stop the sampler and reap its multiprocessing children before hard exit.

  Forked DP workers inherit stdout/stderr. Leaving them alive keeps the
  launcher's tee pipe open even after the evaluator calls os._exit(0).
  """
  preexisting = set(multiprocessing.active_children())
  sampler = None
  try:
    sampler = factory()
    yield sampler
  finally:
    children = [
        child for child in multiprocessing.active_children()
        if child not in preexisting
    ]
    _stop_sampler(
        sampler, children, stop_timeout_seconds, join_timeout_seconds
    )
