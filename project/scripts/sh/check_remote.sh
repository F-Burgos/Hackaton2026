#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-${REMOTE:-}}"
REMOTE_DIR="${2:-~/Hackaton2026}"

if [[ -z "${REMOTE}" ]]; then
  echo "Usage: REMOTE=user@private-host $0 [remote] [remote_dir]" >&2
  echo "Remote host/user are intentionally not versioned." >&2
  exit 2
fi

ssh "${REMOTE}" "cd ${REMOTE_DIR} && \
  echo '== Remote Git ==' && git status --short --branch && git log --oneline --decorate -3 && \
  echo && echo '== Remote GPU ==' && (nvidia-smi || true) && \
  echo && echo '== User Processes ==' && ps -u \"\$USER\" -o pid,ppid,stat,etime,pcpu,pmem,cmd --sort=-pcpu | head -30"
