#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-5001}"
PID_FILE="${ROOT_DIR}/.ui.pid"
LOG_FILE="${ROOT_DIR}/.ui.log"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" >/dev/null 2>&1; then
    echo "UI already running (pid ${existing_pid})."
    echo "URL: http://127.0.0.1:${PORT}"
    exit 0
  fi
fi

nohup env PORT="${PORT}" python "${ROOT_DIR}/app.py" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"
echo "UI started (pid $!)."
echo "URL: http://127.0.0.1:${PORT}"
