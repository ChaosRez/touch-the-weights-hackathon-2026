"""Guards for the live rollout path (regressions only a live eval would otherwise catch).

1. Each toolset module must be runnable as an MCP server: the v1 launcher serves a toolset by
   running its module as ``python -m <module>`` and waiting for it to write its bound port to
   ``$MCP_PORT_FILE`` (``ServerBase.run`` -> ``_serve``). Without an ``if __name__ ==
   "__main__": <Toolset>.run()`` block the process imports and exits, no port is ever written,
   and every live rollout dies with ``server did not report its port`` — invisible to the
   offline ``Task.score`` tests. These tests actually spawn the module and assert a port lands.

2. Every instance must carry the answer-format system prompt, or exact-match scoring is
   unsolvable in practice (a model answers "13,923,977 cents", gold is "13923977").
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig
from alien_api_env.vf.taskset import SYSTEM_PROMPT


@pytest.mark.parametrize(
    "module",
    ["alien_api_env.vf.tools.crm", "alien_api_env.vf.tools.wiki"],
)
def test_toolset_module_serves_and_reports_port(module: str, tmp_path) -> None:
    port_file = tmp_path / "port"
    env = {**os.environ, "VF_CONFIG": "{}", "MCP_PORT_FILE": str(port_file)}
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if port_file.exists() and port_file.read_text().strip().isdigit():
                break
            if proc.poll() is not None:  # exited before writing a port
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"{module} exited (code {proc.returncode}) without serving:\n{out}")
            time.sleep(0.5)
        assert port_file.exists() and port_file.read_text().strip().isdigit(), (
            f"{module} did not report its MCP port within 30s"
        )
        assert int(port_file.read_text().strip()) > 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_instances_carry_answer_format_system_prompt() -> None:
    ts = AlienApiTaskset(AlienApiTasksetConfig(id="alien-api"))
    for task in ts.select(12):
        assert task.data.system_prompt == SYSTEM_PROMPT
    # The prompt documents the answer channel (including the escalate token) without
    # revealing any of the feedbacker's conventions.
    assert "submit_answer" in SYSTEM_PROMPT
    assert "escalate" in SYSTEM_PROMPT  # channel documentation; *when* is the preference
    for banned in (
        "billing",
        "shipping",
        "fiscal",
        "cents",
        "dollars",
        "aggregate",
        "reserved",
        "stale",
        "deprecated",
    ):
        assert banned not in SYSTEM_PROMPT.lower(), (
            f"system prompt leaks the preference hint {banned!r}"
        )
