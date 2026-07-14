#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-felipeiburgos@titae.inf.udec.cl}"
REMOTE_DIR="${2:-~/Hackaton2026}"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && \
  echo '== Remote Git ==' && git status --short --branch && git log --oneline --decorate -3 && \
  echo && echo '== Remote GPU ==' && (nvidia-smi || true) && \
  echo && echo '== User Processes ==' && ps -u \"\$USER\" -o pid,ppid,stat,etime,pcpu,pmem,cmd --sort=-pcpu | head -30"
