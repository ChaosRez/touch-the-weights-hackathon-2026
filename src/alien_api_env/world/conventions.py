"""Shared deterministic convention helpers: the single seam tools, templates, and the
label compiler all resolve values through, so they cannot drift from one another.

Two groups:

- **World/tool behaviors** (carried over from the v2 ``questions.resolve_*`` seam): the
  opportunity lookup key, the fresh regional MRR aggregate, and the newest-account
  helper the stale-report behavior needs. ``vf/tools/crm.py`` calls these.
- **Presentation renderers** for the preference layer: money under a
  ``(unit, rounding)`` choice, ISO dates. The instance templates enumerate defensible
  answers through these, and only these, so the accepted label and the strings a
  convention-following agent would produce are byte-identical.

All pure, synchronous, sub-second.
"""

from __future__ import annotations

from datetime import UTC, datetime

from alien_api_env.world.models import Account, Behavior, World

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

# --------------------------------------------------------------------------- #
# World/tool behaviors (the anti-drift seam the tools realize).                 #
# --------------------------------------------------------------------------- #


def fiscal_quarter(close_date: int, fiscal_year_start_month: int) -> int:
    """Fiscal quarter 1..4 for a Unix ``close_date`` under a fiscal-year start month."""
    month = datetime.fromtimestamp(close_date, tz=UTC).month
    return ((month - fiscal_year_start_month) % 12) // 3 + 1


def resolve_opportunity_key(opp_id: str, behavior: Behavior) -> str:
    """The working API lookup key for an opportunity (the required prefix applied)."""
    return f"{behavior.opportunity_lookup_prefix}{opp_id}"


def resolve_report_mrr(world: World, region: int) -> int:
    """Current (fresh) total mrr for a region, in cents."""
    return sum(a.mrr for a in world.accounts.values() if a.region == region)


def newest_account_in_region(world: World, region: int) -> Account | None:
    """The region's newest account (max id): what the stale report silently excludes."""
    in_region = [a for a in world.accounts.values() if a.region == region]
    return max(in_region, key=lambda a: a.id) if in_region else None


# --------------------------------------------------------------------------- #
# Presentation renderers (the preference layer's format seam).                  #
# --------------------------------------------------------------------------- #


def _half_up(value: int, quantum: int) -> int:
    """Round ``value`` to the nearest multiple of ``quantum``, halves up. Deterministic
    (no banker's rounding), non-negative inputs only — every rendered figure is a sum of
    non-negative cents."""
    return ((value + quantum // 2) // quantum) * quantum


def render_money(cents: int, unit: str, rounding: str = "exact") -> str:
    """A monetary answer under a ``money_unit`` x ``money_rounding`` choice.

    ``cents`` -> the bare integer of cents; ``dollars`` -> whole dollars, halves up.
    ``thousands`` rounds the resulting figure to the nearest thousand of its unit.
    Always a bare integer string (separators/currency symbols are prose, not values).
    """
    if unit == "cents":
        value = cents
    elif unit == "dollars":
        value = _half_up(cents, 100) // 100
    else:  # pragma: no cover - guarded by schema validation
        raise ValueError(f"unknown money unit {unit!r}")
    if rounding == "thousands":
        value = _half_up(value, 1000)
    elif rounding != "exact":  # pragma: no cover - guarded by schema validation
        raise ValueError(f"unknown money rounding {rounding!r}")
    return str(value)


def iso_date(ts: int) -> str:
    """The UTC calendar date of a Unix timestamp, ISO-8601 (``2025-03-14``)."""
    return datetime.fromtimestamp(ts, tz=UTC).date().isoformat()
