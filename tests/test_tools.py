"""Tool surface (v4): envelope well-formedness, stored-world round-trips, hydration
determinism, artifact sizing, and the behavior-block realizations at the tool boundary.

Tools are exercised in-process (call the methods directly after ``setup_task``), not
through MCP — the MCP dispatch itself is the network-gated live probe.
"""

from __future__ import annotations

import json

import verifiers.v1 as vf

from alien_api_env.vf import AlienApiData, CrmToolset, WikiToolset
from alien_api_env.vf.tools.envelopes import ARTIFACT_VERBOSITY, count_tokens
from alien_api_env.world import conventions as C
from alien_api_env.world.store import load_world


async def _crm() -> CrmToolset:
    ts = CrmToolset(vf.ToolsetConfig())
    await ts.setup_task(AlienApiData())  # defaults: the one committed world
    return ts


async def _wiki() -> WikiToolset:
    ts = WikiToolset(vf.ToolsetConfig())
    await ts.setup_task(AlienApiData())
    return ts


def _assert_ok(env: dict) -> dict:
    assert env["ok"] is True, env
    assert "data" in env and "meta" in env
    return env


async def test_get_account_round_trips_real_record() -> None:
    world = load_world()
    crm = await _crm()
    aid = next(iter(world.accounts))
    env = _assert_ok(await crm.get_account(aid))
    acc = world.accounts[aid]
    assert env["data"]["id"] == acc.id
    assert env["data"]["mrr"] == acc.mrr
    assert env["data"]["region"] == acc.region


async def test_get_order_round_trips_status_code() -> None:
    world = load_world()
    crm = await _crm()
    oid = next(iter(world.orders))
    env = _assert_ok(await crm.get_order(oid))
    assert env["data"]["status_code"] == world.orders[oid].status_code


async def test_read_page_round_trips_prose() -> None:
    world = load_world()
    wiki = await _wiki()
    env = _assert_ok(await wiki.read_page("sop-revenue-recognition"))
    assert env["data"]["body"] == world.sop.pages["sop-revenue-recognition"].body


async def test_unknown_ids_return_error_envelopes() -> None:
    crm = await _crm()
    for env in (await crm.get_account("acc-does-not-exist"), await crm.get_order("ord-x")):
        assert env["ok"] is False
        assert env["error"]["code"] == "not_found"
    wiki = await _wiki()
    env = await wiki.read_page("sop-nope")
    assert env["ok"] is False and env["error"]["code"] == "not_found"


async def test_query_report_rejects_unknown_metric() -> None:
    crm = await _crm()
    env = await crm.query_report("revenue", 0)
    assert env["ok"] is False and env["error"]["code"] == "unknown_metric"


async def test_search_cap_realized_and_aggregate_honest() -> None:
    world = load_world()
    crm = await _crm()
    # region 0 overflows the cap by construction; the aggregate is the honest count.
    true_region0 = sum(1 for a in world.accounts.values() if a.region == 0)
    env = _assert_ok(await crm.count_accounts(region=0))
    assert env["data"]["count"] == true_region0
    search = await crm.search_accounts(region=0)
    assert search["meta"]["count"] == world.behavior.search_cap < true_region0


async def test_setup_task_determinism_byte_identical() -> None:
    a = await _crm()
    b = await _crm()
    for call in (
        lambda t: t.search_accounts(region=0),
        lambda t: t.list_invoices(segment="enterprise"),
        lambda t: t.list_opportunities(region=1),
        lambda t: t.list_products(),
    ):
        ea, eb = await call(a), await call(b)
        assert json.dumps(ea, sort_keys=True, default=str) == json.dumps(eb, sort_keys=True, default=str)


async def test_representative_instance_lands_in_artifact_band() -> None:
    crm = await _crm()
    seq = [
        await crm.search_accounts(region=0),
        await crm.list_invoices(segment="enterprise"),
        await crm.list_opportunities(region=1),
        await crm.list_products(),
    ]
    total = sum(count_tokens(json.dumps(env, default=str)) for env in seq)
    assert 0.4 * ARTIFACT_VERBOSITY <= total <= 2.5 * ARTIFACT_VERBOSITY, total


async def test_read_changelog_and_list_spaces() -> None:
    world = load_world()
    wiki = await _wiki()
    spaces = _assert_ok(await wiki.list_spaces())["data"]["spaces"]
    assert "Finance" in spaces and "Ops" in spaces
    changelog = _assert_ok(await wiki.read_changelog())["data"]
    assert len(changelog) == len(world.sop.changelog)


# --- behavior realizations at the tool boundary ---


async def test_opportunity_prefix_gate_realized() -> None:
    world = load_world()
    crm = await _crm()
    opp = next(iter(world.opportunities.values()))
    prefixed = C.resolve_opportunity_key(opp.id, world.behavior)
    assert (await crm.get_opportunity(opp.id))["ok"] is False  # bare id -> not_found
    assert (await crm.get_opportunity(prefixed))["ok"] is True  # prefixed key works


async def test_no_tool_surfaces_the_lookup_key() -> None:
    crm = await _crm()
    rows = (await crm.list_opportunities())["data"]
    assert rows and all("lookup_key" not in row for row in rows)


async def test_deprecated_inventory_endpoint_realized() -> None:
    world = load_world()
    crm = await _crm()
    b = world.behavior
    env = await crm.get_inventory(b.deprecated_inventory_endpoint)
    assert env["ok"] is False and env["meta"]["status"] == 404
    assert (await crm.get_inventory(b.working_inventory_endpoint))["ok"] is True


async def test_report_goes_stale_on_retry() -> None:
    crm = await _crm()
    first = _assert_ok(await crm.query_report("mrr", 1))["data"]["value"]
    second = _assert_ok(await crm.query_report("mrr", 1))["data"]["value"]
    assert second < first  # the retry silently excludes the newest account
