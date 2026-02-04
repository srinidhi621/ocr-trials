#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/.ui.pid"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" >/dev/null 2>&1; then
    kill "${existing_pid}" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "${existing_pid}" >/dev/null 2>&1; then
      kill -9 "${existing_pid}" >/dev/null 2>&1 || true
    fi
    echo "Stopped UI (pid ${existing_pid})."
  fi
  rm -f "${PID_FILE}"
fi

# Best-effort cleanup for any lingering UI processes
pkill -f "/scb_trials/app.py" >/dev/null 2>&1 || true
pkill -f "PORT=5001 python app.py" >/dev/null 2>&1 || true
pkill -f "python app.py" >/dev/null 2>&1 || true
echo "UI shutdown complete."
