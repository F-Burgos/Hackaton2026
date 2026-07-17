#!/usr/bin/env bash

resolve_project_python() {
  if [[ -x ".venv/bin/python" ]]; then
    printf '%s\n' ".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    command -v python
  fi
}
