"""The SOP / Notion wiki toolset (verifiers-v1 MCP toolset).

Reads the stored ``world.sop`` knowledge base verbatim. Governed page pairs both claim
currency (no supersession metadata exists), so which page governs a process is a learned
convention, uncontradicted in-instance.

Same per-task world hydration as the CRM toolset; stateless (no ``vf.State`` subclass).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from alien_api_env.vf.tools.envelopes import err, ok

if TYPE_CHECKING:  # avoid a runtime import cycle
    from alien_api_env.vf.taskset import AlienApiData


class WikiToolset(vf.Toolset[vf.ToolsetConfig]):
    """The alien-api SOP/wiki read surface, served over MCP. Stateless."""

    async def setup_task(self, task: AlienApiData) -> None:
        from alien_api_env.world.store import load_world

        self._world = await asyncio.to_thread(load_world, task.world, task.worlds_root)

    def _all_pages(self) -> dict[str, Any]:
        sop = self._world.sop
        return {**dict(sop.pages), **dict(sop.runbooks)}

    @vf.tool
    async def list_spaces(self) -> dict[str, Any]:
        """List the knowledge-base spaces (Finance, Sales, Ops, Support)."""
        spaces = sorted({p.space for p in self._all_pages().values()})
        return ok({"spaces": spaces})

    @vf.tool
    async def search_wiki(self, query: str) -> dict[str, Any]:
        """Search SOP pages and runbooks; return id, title, space, and a snippet per match.

        Args:
            query: a case-insensitive substring matched against page id, title, and body.
        """
        q = query.lower()
        hits = []
        for page in self._all_pages().values():
            hay = f"{page.id}\n{page.title}\n{page.body}".lower()
            if q in hay:
                idx = page.body.lower().find(q)
                start = max(0, idx - 80) if idx >= 0 else 0
                snippet = page.body[start : start + 200].strip()
                hits.append(
                    {"id": page.id, "title": page.title, "space": page.space, "snippet": snippet}
                )
        return ok(hits, count=len(hits))

    @vf.tool
    async def read_page(self, page_id: str) -> dict[str, Any]:
        """Read one SOP page or runbook: its full prose body.

        Args:
            page_id: the page id, for example ``sop-revenue-recognition``.
        """
        page = self._all_pages().get(page_id)
        if page is None:
            return err("not_found", f"no page with id {page_id!r}")
        data = {"id": page.id, "space": page.space, "title": page.title, "body": page.body}
        return ok(data)

    @vf.tool
    async def read_changelog(self) -> dict[str, Any]:
        """Read the knowledge base's deploy/changelog notes, newest concerns last."""
        entries = [asdict(e) for e in self._world.sop.changelog]
        return ok(entries, count=len(entries))


if __name__ == "__main__":  # MCP server entrypoint: `python -m alien_api_env.vf.tools.wiki`
    # See CrmToolset: the v1 launcher runs each toolset module as `python -m <module>` and
    # `ServerBase.run()` reads `VF_CONFIG` and binds the MCP server.
    WikiToolset.run()
