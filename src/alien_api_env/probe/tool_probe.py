"""Tool-registration probe: which v1 mechanism registers tools, and does the harness
actually dispatch to them?

## Verdict (settled from the verifiers 0.2.2.dev29 source, not a guess)

v1 has ONE tool-registration mechanism, and the framework picks it — there is no menu:

- **`@vf.tool` on a `vf.Toolset` method** — THE mechanism. `Toolset._register` walks
  `discover_decorated(self, "tool")` and registers each as an MCP tool (method name →
  tool name, docstring → description). A `Task`/`Taskset` lists its toolset classes in
  its `tools` classvar. This is what `CrmToolset` / `WikiToolset` use.
- **Plain functions (v0 `add_tool(fn, args_to_skip=...)`)** — GONE in v1. There is no
  `StatefulToolEnv`, no `add_tool`, no `update_tool_args`.
- **`mcp_urls`** — NOT an authoring mechanism. It is the wiring the framework hands a
  harness at launch (the live tool-server URLs); external MCP servers attach via a
  toolset's `ToolsetConfig.url`, not by the environment author registering URLs.

The remaining empirical question is **dispatch**: does a harness advertise the toolset's
tools to the model, call them, and return their payloads into the trace? That requires a
real rollout, which in v1 always proxies model calls through the interception server to a
configured model endpoint — there is no fully offline rollout. So:

- `verify_mechanism()` (no network) statically confirms `@vf.tool` discovery and MCP
  registration. This runs in CI.
- `run_live_probe()` (needs an endpoint) runs one real rollout under the `null` harness
  (SUPPORTS_MCP, no container) against an OpenAI-compatible endpoint and asserts a tool
  was dispatched. Gated on env vars; skipped when they are absent.

Run: `python -m alien_api_env.probe.tool_probe`
Live rollout also needs: VF_PROBE_BASE_URL, VF_PROBE_API_KEY, VF_PROBE_MODEL.
"""

from __future__ import annotations

import asyncio
import os

import verifiers.v1 as vf
from verifiers.v1.decorators import discover_decorated

from alien_api_env.vf.tools import CrmToolset, WikiToolset


def verify_mechanism() -> dict[str, object]:
    """Statically confirm the `@vf.tool` + `Toolset` + MCP mechanism (no network).

    Returns a verdict dict; raises AssertionError if the mechanism is not intact.
    """
    from alien_api_env.vf.taskset import AlienApiTask

    toolset = CrmToolset(vf.ToolsetConfig())
    discovered = [getattr(fn, "tool_name", None) or fn.__name__ for fn in discover_decorated(toolset, "tool")]

    verdict = {
        "mechanism": "vf.Toolset + @vf.tool (MCP)",
        "toolset_class": CrmToolset.__name__,
        "config_cls": CrmToolset._config_cls().__name__,
        "server_name": toolset.server_name,
        "registered_tools": discovered,
        "task_wires_toolsets": [c.__name__ for c in AlienApiTask.tools],
        "plain_function_add_tool": "absent in v1 (no StatefulToolEnv/add_tool)",
        "mcp_urls": "framework-supplied at launch, not an authoring surface",
        "harness_requires": "SUPPORTS_MCP == True (e.g. null, bash)",
    }

    assert "get_account" in discovered, f"tools not discovered: {discovered}"
    assert CrmToolset in AlienApiTask.tools, "AlienApiTask does not wire CrmToolset"
    assert WikiToolset in AlienApiTask.tools, "AlienApiTask does not wire WikiToolset"
    return verdict


async def run_live_probe() -> dict[str, object] | None:
    """Run one real rollout under the `null` harness and assert a tool was dispatched.

    Returns a verdict dict, or None if the endpoint env vars are not set (skip).
    Requires: VF_PROBE_BASE_URL, VF_PROBE_API_KEY, VF_PROBE_MODEL. A v1 rollout proxies
    the model call through the interception server to this endpoint, so a live model is
    unavoidable — there is no offline rollout.
    """
    base_url = os.environ.get("VF_PROBE_BASE_URL")
    api_key = os.environ.get("VF_PROBE_API_KEY")
    model = os.environ.get("VF_PROBE_MODEL")
    if not (base_url and api_key and model):
        return None

    # Imported lazily so the module (and CI's static check) never depends on the
    # heavier run surface.
    from verifiers.v1 import (
        AgentConfig,
        ClientConfig,
        ModelContext,
        SamplingConfig,
    )
    from verifiers.v1.clients import resolve_client
    from verifiers.v1.envs.single_agent import SingleAgentEnv, SingleAgentEnvConfig

    # A prompt that forces a real tool call so dispatch is exercised, not just chat.
    tool_prompt = (
        "Call the `count_accounts` tool with no arguments, then reply with exactly the "
        "returned count and nothing else."
    )

    class _ProbeTaskset(vf.Taskset[vf.Task, vf.TasksetConfig]):
        def load(self):
            from alien_api_env.vf.taskset import AlienApiData, AlienApiTask

            yield AlienApiTask(
                AlienApiData(idx=0, name="probe", prompt=tool_prompt, budget=4),
                self.config.task,
            )

    # Pin the null harness (SUPPORTS_MCP, no container) and the run's endpoint.
    env_config = SingleAgentEnvConfig(
        taskset=vf.TasksetConfig(id="alien-api"),
        agent=AgentConfig(harness={"id": "null"}, max_turns=4),
    )
    # Build the env directly rather than via the id loader so the probe runs from source.
    env = SingleAgentEnv(env_config)
    env.taskset = _ProbeTaskset(vf.TasksetConfig(id="alien-api"))

    client = resolve_client(ClientConfig(base_url=base_url, api_key=api_key))
    ctx = ModelContext(client=client, model=model, sampling=SamplingConfig(temperature=0.0))

    async with env.serving():
        task = env.taskset.select(1)[0]
        episode = await env.run_episode(task, ctx)

    trace = episode.traces[0]
    tool_msgs = [m for m in trace.tool_messages]
    dispatched = len(tool_msgs) > 0
    return {
        "endpoint": base_url,
        "model": model,
        "num_turns": trace.num_turns,
        "tool_messages": len(tool_msgs),
        "dispatched": dispatched,
        "reward": trace.reward,
        "ok": trace.ok,
    }


def _print_verdict(title: str, verdict: dict[str, object] | None) -> None:
    print(f"\n=== {title} ===")
    if verdict is None:
        print("SKIPPED (set VF_PROBE_BASE_URL / VF_PROBE_API_KEY / VF_PROBE_MODEL to run)")
        return
    width = max(len(k) for k in verdict)
    for key, value in verdict.items():
        print(f"  {key:<{width}} : {value}")


def main() -> int:
    static_verdict = verify_mechanism()
    _print_verdict("static mechanism verdict (CI-safe)", static_verdict)

    live_verdict = asyncio.run(run_live_probe())
    _print_verdict("live rollout dispatch verdict", live_verdict)

    if live_verdict is not None and not live_verdict.get("dispatched"):
        print("\nFAIL: live rollout ran but no tool was dispatched.")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
