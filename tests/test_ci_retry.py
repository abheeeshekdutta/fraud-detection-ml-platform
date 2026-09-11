from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("succeed_on", "expected_calls", "expected_delays", "exit_code"),
    [(1, 1, [], 0), (3, 3, ["5", "10"], 0), (9, 4, ["5", "10", "20"], 7)],
)
def test_dependency_retries_are_bounded_and_preserve_failure(
    tmp_path,
    succeed_on,
    expected_calls,
    expected_delays,
    exit_code,
) -> None:
    command = tmp_path / "dependency.sh"
    command.write_text(
        "#!/bin/bash\n"
        "count=0\n"
        'if [[ -f "$COUNT_FILE" ]]; then read -r count < "$COUNT_FILE"; fi\n'
        "count=$((count + 1))\n"
        'echo "$count" > "$COUNT_FILE"\n'
        "if (( count < SUCCEED_ON )); then exit 7; fi\n"
    )
    sleeper = tmp_path / "sleep"
    sleeper.write_text('#!/bin/bash\necho "$1" >> "$SLEEP_LOG"\n')
    sleeper.chmod(0o755)
    count_file = tmp_path / "count"
    sleep_log = tmp_path / "delays"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci_retry.sh"), "bash", str(command)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "COUNT_FILE": str(count_file),
            "SLEEP_LOG": str(sleep_log),
            "SUCCEED_ON": str(succeed_on),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code
    assert int(count_file.read_text()) == expected_calls
    assert (sleep_log.read_text().splitlines() if sleep_log.exists() else []) == expected_delays
