"""The CRM/ERP query toolset (verifiers-v1 MCP toolset).

Hydrates the stored world once per task in ``setup_task`` (off the event loop) and serves
it through ``@vf.tool`` methods. The world's **behavior block** is *realized here*:

- ``search_cap`` : ``search_accounts`` silently caps at this many matches; the honest
  count is recoverable via the ``count_accounts`` aggregate.
- ``opportunity_lookup_prefix`` : ``get_opportunity`` needs the prefixed key; the bare id
  returns ``not_found``.
- ``report_stale_on_retry`` : the *first* ``query_report`` call is fresh/honest; a retry
  within the window silently returns the stale aggregate (excludes the region's newest
  account).
- ``region_filter_exclusive_invoices`` : the ``region`` filter is inclusive on accounts
  but **exclusive** on ``list_invoices``; the honest set is recovered by pulling
  unfiltered and filtering client-side on each invoice's own ``region``.
- ``deprecated_inventory_endpoint`` / ``working_inventory_endpoint`` : the documented
  inventory route 404s; the working route is undocumented.

No dedicated authoritative references exist for the latent conventions (no fiscal
calendar, no status reference, no endpoint index, no supersession metadata): a cold
tool-user has no in-instance basis to distrust the drifted SOP; the conventions are
learned from the feedbacker's corrections. Every shared computation routes through
``world/conventions.py`` (the anti-drift seam shared with the build-time templates), so
the tools and the stored labels cannot drift.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from alien_api_env.vf.tools.envelopes import (
    ARTIFACT_VERBOSITY,
    LIST_CALLS_PER_INSTANCE,
    err,
    ok,
    pad_to_tokens,
)
from alien_api_env.world import conventions as C

if TYPE_CHECKING:  # avoid a runtime import cycle (taskset imports this module)
    from alien_api_env.vf.taskset import AlienApiData


def _rec(obj: Any) -> dict[str, Any]:
    """A frozen record dataclass as a plain JSON-safe dict."""
    return asdict(obj)


class CrmToolset(vf.Toolset[vf.ToolsetConfig]):
    """The alien-api CRM/ERP read surface, served over MCP.

    Stateless in the verifiers sense (no ``vf.State`` subclass, so no per-call state-channel
    sync): the immutable world is a plain instance attribute built once in ``setup_task``.
    A small per-instance call counter for the stale-report behavior is likewise a plain
    attribute, reset per task — not synced state and not a scratchpad/notepad (memory is a
    property of the system under test, excluded by design).
    """

    async def setup_task(self, task: AlienApiData) -> None:
        """Hydrate the stored world off the event loop (cached per process) and stash it."""
        from alien_api_env.world.store import load_world

        self._world = await asyncio.to_thread(load_world, task.world, task.worlds_root)
        self._artifact_target = getattr(task, "artifact_verbosity", ARTIFACT_VERBOSITY)
        self._report_calls: dict[tuple[str, int], int] = {}

    async def setup(self) -> None:
        # Task-agnostic sibling; the counter is (re)initialized per task in setup_task.
        self._report_calls = {}

    @property
    def _list_target(self) -> int:
        return getattr(self, "_artifact_target", ARTIFACT_VERBOSITY) // LIST_CALLS_PER_INSTANCE

    # ----------------------------------------------------------------- accounts

    def _account_matches(
        self, acc: Any, region: int | None, segment: int | None, campaign: str | None
    ) -> bool:
        if region is not None and acc.region != region:
            return False
        if segment is not None and acc.segment != segment:
            return False
        if campaign is not None and acc.churn_campaign != campaign:
            return False
        return True

    @vf.tool
    async def search_accounts(
        self,
        region: int | None = None,
        segment: str | None = None,
        campaign: str | None = None,
    ) -> dict[str, Any]:
        """Search accounts by an AND-combined filter and return the matching records.

        Args:
            region: restrict to this integer sales region.
            segment: restrict to this account tier (enterprise, midmarket, or smb).
            campaign: restrict to accounts that churned after this campaign.
        """
        matches = [
            a
            for a in self._world.accounts.values()
            if self._account_matches(a, region, segment, campaign)
        ]
        matches = matches[: self._world.behavior.search_cap]  # silent cap, no cursor, no indicator
        env = ok([_rec(a) for a in matches], count=len(matches))
        return pad_to_tokens(env, self._list_target)

    @vf.tool
    async def count_accounts(
        self,
        region: int | None = None,
        segment: str | None = None,
        campaign: str | None = None,
    ) -> dict[str, Any]:
        """Return the exact count of accounts matching an AND-combined filter.

        A server-side aggregate that is not subject to the ``search_accounts`` result cap.

        Args:
            region: restrict to this integer sales region.
            segment: restrict to this account tier (enterprise, midmarket, or smb).
            campaign: restrict to accounts that churned after this campaign.
        """
        n = sum(
            1
            for a in self._world.accounts.values()
            if self._account_matches(a, region, segment, campaign)
        )
        return ok({"count": n})

    @vf.tool
    async def get_account(self, id: str) -> dict[str, Any]:
        """Return one account record by id, or a ``not_found`` error.

        Args:
            id: the account id, for example ``acc-00042``.
        """
        acc = self._world.accounts.get(id)
        if acc is None:
            return err("not_found", f"no account with id {id!r}")
        return ok(_rec(acc))

    # ------------------------------------------------------------ opportunities

    @vf.tool
    async def list_opportunities(
        self,
        region: int | None = None,
        stage: str | None = None,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        """List opportunities matching an AND-combined filter.

        Args:
            region: restrict to this integer sales region.
            stage: restrict to this stage (open, won, or lost).
            account_id: restrict to opportunities on this account.
        """
        out = []
        for opp in self._world.opportunities.values():
            if region is not None and opp.region != region:
                continue
            if stage is not None and opp.stage != stage:
                continue
            if account_id is not None and opp.account_id != account_id:
                continue
            # No tool surfaces the working lookup key; the bare record id is all that is
            # exposed and the required prefix is a learned convention.
            out.append(_rec(opp))
        env = ok(out, count=len(out))
        return pad_to_tokens(env, self._list_target)

    @vf.tool
    async def get_opportunity(self, key: str) -> dict[str, Any]:
        """Return one opportunity by its API lookup key, or a ``not_found`` error.

        Args:
            key: the opportunity lookup key (the id with its required prefix, if any).
        """
        b = self._world.behavior
        for opp in self._world.opportunities.values():
            if key == C.resolve_opportunity_key(opp.id, b):  # prefix-gated lookup
                return ok(_rec(opp))
        return err("not_found", f"no opportunity for lookup key {key!r}")

    # -------------------------------------------------------------------- orders

    @vf.tool
    async def get_order(self, id: str) -> dict[str, Any]:
        """Return one order record by id, including its opaque integer ``status_code``.

        Args:
            id: the order id, for example ``ord-00042``.
        """
        order = self._world.orders.get(id)
        if order is None:
            return err("not_found", f"no order with id {id!r}")
        return ok(_rec(order))

    # `get_order` returns only the opaque integer status code; translating it is a
    # learned convention (no status-reference tool exists).

    # ------------------------------------------------------------------ invoices

    @vf.tool
    async def list_invoices(
        self, region: int | None = None, segment: str | None = None
    ) -> dict[str, Any]:
        """List invoices matching an AND-combined filter.

        Args:
            region: restrict by region using the endpoint's region filter.
            segment: restrict by the owning account's tier (enterprise, midmarket, smb).
        """
        exclusive = self._world.behavior.region_filter_exclusive_invoices
        accounts = self._world.accounts
        out = []
        for inv in self._world.invoices.values():
            if region is not None:
                in_region = inv.region == region
                # the region filter is exclusive on invoices (inclusive elsewhere), so
                # it returns invoices NOT in the named region.
                keep = (not in_region) if exclusive else in_region
                if not keep:
                    continue
            if segment is not None and accounts[inv.account_id].segment != segment:
                continue
            out.append(_rec(inv))
        env = ok(out, count=len(out))
        return pad_to_tokens(env, self._list_target)

    # ------------------------------------------------------------------- reports

    @vf.tool
    async def query_report(self, metric: str, region: int) -> dict[str, Any]:
        """Return a precomputed aggregate figure for a region.

        Supported metric: ``mrr`` (current total monthly recurring revenue for the region).

        Args:
            metric: the aggregate to compute (currently ``mrr``).
            region: the integer sales region.
        """
        if metric != "mrr":
            return err("unknown_metric", f"unsupported metric {metric!r}")
        key = (metric, region)
        n = self._report_calls.get(key, 0) + 1
        self._report_calls[key] = n
        fresh = C.resolve_report_mrr(self._world, region)  # honest, through the shared seam
        if n == 1 or not self._world.behavior.report_stale_on_retry:
            return ok({"metric": metric, "region": region, "value": fresh})
        # a retry within the window silently returns the stale cached aggregate
        # (computed before the region's newest account), never an error, never a flag.
        newest = C.newest_account_in_region(self._world, region)
        stale = fresh - (newest.mrr if newest else 0)
        return ok({"metric": metric, "region": region, "value": stale})

    # `list_opportunities` returns raw `close_date`s; the fiscal-year start month is a
    # learned convention (no fiscal-calendar tool exists).

    # ----------------------------------------------------------------- inventory

    # The working inventory route is a learned convention (no endpoint index exists); a
    # cold agent following the data-dictionary SOP reaches for the deprecated route.

    @vf.tool
    async def get_inventory(self, endpoint: str) -> dict[str, Any]:
        """Read current inventory levels from the named endpoint.

        The endpoint the data-dictionary SOP documents may be deprecated and return 404. Which
        route actually serves is a learned convention (no reference lists it); a wrong route
        returns a ``not_found`` 404.

        Args:
            endpoint: the inventory endpoint route to read from.
        """
        b = self._world.behavior
        deprecated, working = b.deprecated_inventory_endpoint, b.working_inventory_endpoint
        if endpoint == working:
            env = ok([_rec(it) for it in self._world.inventory.values()], endpoint=endpoint)
            return pad_to_tokens(env, self._list_target)
        if endpoint == deprecated:  # deprecated route 404s, no hint of the working one
            return err("not_found", f"endpoint {endpoint!r} is not available", status=404)
        return err("not_found", f"unknown endpoint {endpoint!r}", status=404)

    # -------------------------------------------------------------------- peers

    @vf.tool
    async def list_products(self) -> dict[str, Any]:
        """List every product in the catalog."""
        env = ok([_rec(p) for p in self._world.products.values()], count=len(self._world.products))
        return pad_to_tokens(env, self._list_target)

    @vf.tool
    async def list_tickets(self, account_id: str | None = None) -> dict[str, Any]:
        """List support tickets, optionally scoped to one account.

        Args:
            account_id: restrict to tickets on this account.
        """
        out = [
            _rec(t)
            for t in self._world.tickets.values()
            if account_id is None or t.account_id == account_id
        ]
        env = ok(out, count=len(out))
        return pad_to_tokens(env, self._list_target)


if __name__ == "__main__":  # MCP server entrypoint: `python -m alien_api_env.vf.tools.crm`
    # The v1 launcher (`verifiers/v1/mcp/launch.py`) serves each toolset by running its module
    # as `python -m <module>`; `ServerBase.run()` reads `VF_CONFIG` and binds the MCP server.
    CrmToolset.run()
