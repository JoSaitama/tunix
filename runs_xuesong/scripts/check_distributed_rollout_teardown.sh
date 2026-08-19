#!/usr/bin/env bash
set -uo pipefail

# Read-only classifier for the dual-worker distributed-vLLM shutdown path.
# Usage:
#   bash runs_xuesong/scripts/check_distributed_rollout_teardown.sh \
#     "$TRANSPORT_LOG_ROOT"
# The argument may also be the launcher.nohup.log file itself.

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "Usage: $0 LOG_ROOT_OR_LAUNCHER_LOG" >&2
  exit 2
fi

if [[ -d "${target}" ]]; then
  launcher_log="${target%/}/launcher.nohup.log"
else
  launcher_log="${target}"
fi

if [[ ! -r "${launcher_log}" ]]; then
  echo "Launcher log is not readable: ${launcher_log}" >&2
  exit 2
fi

normalized_log="$(mktemp "${TMPDIR:-/tmp}/tunix-teardown-log.XXXXXX")"
trap 'rm -f "${normalized_log}"' EXIT
tr -d '\000' < "${launcher_log}" > "${normalized_log}"

echo "=== INPUT ==="
echo "LAUNCHER_LOG=${launcher_log}"

echo "=== JAX IDENTITIES ==="
grep -E 'JAX_DISTRIBUTED_IDENTITY' "${normalized_log}" | tail -n 4 || true

echo "=== CORE ROLLOUT EVIDENCE ==="
grep -E \
  'listener is ready|Accepted distributed rollout message.*update_params|Accepted distributed rollout message.*generate_request|Frozen rollout diagnostic summary|Accepted distributed rollout message.*shutdown' \
  "${normalized_log}" | tail -n 20 || true

echo "=== TEARDOWN PHASES ==="
grep -E 'DISTRIBUTED_TEARDOWN' "${normalized_log}" | tail -n 30 || true

echo "=== PROCESS STATUS ==="
grep -E \
  'HOST=.*STATUS=|LOCAL_STATUS=|REMOTE_SSH_STATUS=|REMOTE_STATUS=' \
  "${normalized_log}" | tail -n 12 || true

has_summary=0
has_shutdown=0
has_cluster_start=0
has_cluster_complete=0
has_rollout_start=0
has_rollout_complete=0
has_finally_complete=0
has_runpy_return=0
has_runpy_system_exit=0
has_runpy_exception=0
has_python_atexit=0
runpy_system_exit_code=""

grep -q 'Frozen rollout diagnostic summary' "${normalized_log}" && has_summary=1
grep -q 'Accepted distributed rollout message.*type shutdown' "${normalized_log}" && has_shutdown=1
grep -q 'phase=rl_cluster_close_start .*process=0' "${normalized_log}" && has_cluster_start=1
grep -q 'phase=rl_cluster_close_complete .*process=0' "${normalized_log}" && has_cluster_complete=1
grep -q 'phase=rollout_close_start .*process=0' "${normalized_log}" && has_rollout_start=1
grep -q 'phase=rollout_close_complete .*process=0' "${normalized_log}" && has_rollout_complete=1
grep -q 'phase=agentic_grpo_finally_complete .*process=0' "${normalized_log}" && has_finally_complete=1
grep -q 'phase=runpy_return .*process=0' "${normalized_log}" && has_runpy_return=1
grep -q 'phase=runpy_system_exit .*process=0' "${normalized_log}" && has_runpy_system_exit=1
grep -q 'phase=runpy_exception .*process=0' "${normalized_log}" && has_runpy_exception=1
grep -q 'phase=python_atexit .*process=0' "${normalized_log}" && has_python_atexit=1
runpy_system_exit_code="$({
  sed -n \
    's/.*phase=runpy_system_exit .*process=0 exit_code=\([^ ]*\).*/\1/p' \
    "${normalized_log}" || true
} | tail -n 1)"

echo "=== CLASSIFICATION ==="
if [[ "${has_summary}" -ne 1 ]]; then
  echo "RESULT=ROLLOUT_OR_REWARD_INCOMPLETE"
elif [[ "${has_cluster_start}" -eq 1 && "${has_cluster_complete}" -ne 1 ]]; then
  echo "RESULT=RL_CLUSTER_CLOSE_FAILED"
elif [[ "${has_rollout_start}" -eq 1 && "${has_rollout_complete}" -ne 1 ]]; then
  echo "RESULT=ROLLOUT_CLOSE_FAILED"
elif [[ "${has_runpy_exception}" -eq 1 ]]; then
  echo "RESULT=PYTHON_EXCEPTION_DURING_OR_AFTER_CLOSE"
elif [[ "${has_runpy_system_exit}" -eq 1 && \
        "${runpy_system_exit_code}" != "0" && \
        "${runpy_system_exit_code}" != "None" ]]; then
  echo "RESULT=RUNPY_NONZERO_SYSTEM_EXIT"
elif [[ "${has_finally_complete}" -eq 1 && \
        "${has_runpy_return}" -ne 1 && \
        "${has_runpy_system_exit}" -ne 1 ]]; then
  echo "RESULT=RUNPY_DID_NOT_RETURN_AFTER_CLEAN_CLOSE"
elif [[ "${has_python_atexit}" -ne 1 && \
        ( "${has_runpy_return}" -eq 1 || \
          "${has_runpy_system_exit}" -eq 1 ) ]]; then
  echo "RESULT=PYTHON_ATEXIT_DID_NOT_COMPLETE"
elif [[ "${has_python_atexit}" -eq 1 ]]; then
  echo "RESULT=PYTHON_MAIN_AND_ATEXIT_COMPLETED"
else
  echo "RESULT=LEGACY_LOG_WITHOUT_TEARDOWN_MARKERS"
fi

echo "SUMMARY_WRITTEN=${has_summary}"
echo "SHUTDOWN_ACCEPTED=${has_shutdown}"
echo "RL_CLUSTER_CLOSE_COMPLETE=${has_cluster_complete}"
echo "ROLLOUT_CLOSE_COMPLETE=${has_rollout_complete}"
echo "RUNPY_RETURNED=${has_runpy_return}"
echo "RUNPY_SYSTEM_EXIT_CODE=${runpy_system_exit_code:-not_seen}"
echo "PYTHON_ATEXIT_REACHED=${has_python_atexit}"

echo "NOTE=This script is read-only and never touches TPU state or checkpoints."
