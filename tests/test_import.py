"""Import + surface tests: the package is a valid verifiers-v1 environment.

No network, no model. Proves the v1 pin resolved and the taskset/task/reward/toolset
surface is well-formed against the installed framework.
"""

from __future__ import annotations

from importlib.metadata import version

import verifiers.v1 as vf
from packaging.version import Version
from verifiers.v1.decorators import discover_decorated

from alien_api_env.vf import (
    AlienApiData,
    AlienApiTask,
    AlienApiTaskset,
    AlienApiTasksetConfig,
    CrmToolset,
    WikiToolset,
)


def test_verifiers_pin_in_range() -> None:
    """The resolved verifiers is a v1 build in [0.2.1, 0.3)."""
    v = Version(version("verifiers"))
    assert Version("0.2.1") <= v < Version("0.3"), f"unexpected verifiers version: {v}"


def test_types_are_v1_subclasses() -> None:
    assert issubclass(AlienApiData, vf.TaskData)
    assert issubclass(AlienApiTask, vf.Task)
    assert issubclass(AlienApiTaskset, vf.Taskset)
    assert issubclass(AlienApiTasksetConfig, vf.TasksetConfig)
    assert issubclass(CrmToolset, vf.Toolset)
    assert issubclass(WikiToolset, vf.Toolset)


def test_taskset_loads_fleet_episodes_with_expected_fields() -> None:
    taskset = AlienApiTaskset(AlienApiTasksetConfig(id="alien-api"))
    tasks = taskset.select(12)
    assert len(tasks) == 12
    data = tasks[0].data
    assert data.world == "kestrel"
    assert data.budget >= 1
    assert data.accepted
    assert data.invoked and data.defensible  # ambiguous by construction
    assert isinstance(data.prompt, str) and data.prompt


def test_task_type_resolves() -> None:
    assert AlienApiTaskset.task_type() is AlienApiTask


def test_reward_and_metrics_are_discoverable() -> None:
    task = AlienApiTaskset(AlienApiTasksetConfig(id="alien-api")).select(1)[0]
    reward_names = [fn.__name__ for fn in discover_decorated(task, "reward")]
    metric_names = [fn.__name__ for fn in discover_decorated(task, "metric")]
    assert "acceptance" in reward_names
    assert {
        "preference_accepted",
        "value_correct",
        "preferences_violated",
        "tool_calls",
        "over_budget",
        "artifact_tokens",
    } <= set(metric_names)


def test_toolsets_register_tools_and_are_wired() -> None:
    crm = CrmToolset(vf.ToolsetConfig())
    crm_tools = {getattr(fn, "tool_name", None) or fn.__name__ for fn in discover_decorated(crm, "tool")}
    assert {"search_accounts", "count_accounts", "get_account", "list_invoices", "query_report"} <= crm_tools

    wiki = WikiToolset(vf.ToolsetConfig())
    wiki_tools = {getattr(fn, "tool_name", None) or fn.__name__ for fn in discover_decorated(wiki, "tool")}
    assert {"list_spaces", "search_wiki", "read_page", "read_changelog"} <= wiki_tools

    # both toolsets are advertised over MCP
    assert CrmToolset in AlienApiTask.tools
    assert WikiToolset in AlienApiTask.tools


def test_toolsets_are_stateless() -> None:
    """No `vf.State` subclass on either toolset — every tool is a pure world read, so no
    per-call state-channel sync (and no scratchpad/notepad; excluded by design)."""
    from verifiers.v1.state import State, state_cls

    assert state_cls(CrmToolset) is State
    assert state_cls(WikiToolset) is State
