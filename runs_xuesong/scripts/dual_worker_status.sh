#!/usr/bin/env bash

# Capture a background process status without allowing `errexit` to terminate
# the caller before the status can be recorded.
tunix_wait_and_capture_status() {
  local result_var="$1"
  local pid="$2"
  local captured_status

  if wait "${pid}"; then
    captured_status=0
  else
    captured_status=$?
  fi
  printf -v "${result_var}" '%s' "${captured_status}"
}

# Prefer the explicit status emitted by the program on the remote worker. The
# SSH transport status is only a fallback when no program status was received.
tunix_resolve_remote_status() {
  local result_var="$1"
  local status_log="$2"
  local ssh_status="$3"
  local program_status

  program_status="$({
    sed -n 's/.*STATUS=\([0-9][0-9]*\).*/\1/p' "${status_log}" 2>/dev/null || true
  } | tail -n 1)"
  if [[ -z "${program_status}" ]]; then
    program_status="${ssh_status}"
  fi
  printf -v "${result_var}" '%s' "${program_status}"
}
