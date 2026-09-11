#!/usr/bin/env bash
# Bounded retries for CI dependency acquisition; do not wrap application tests.
set -uo pipefail

if (( $# == 0 )); then
  echo "Usage: ci_retry.sh command [arguments...]" >&2
  exit 2
fi

for attempt in 1 2 3 4; do
  if "$@"; then
    exit 0
  else
    status=$?
  fi
  if (( attempt == 4 )); then
    echo "Dependency preparation failed after 4 attempts (exit ${status})." >&2
    exit "$status"
  fi
  delay=$((5 * 2 ** (attempt - 1)))
  echo "Attempt ${attempt}/4 failed (exit ${status}); retrying in ${delay}s." >&2
  sleep "$delay"
done
