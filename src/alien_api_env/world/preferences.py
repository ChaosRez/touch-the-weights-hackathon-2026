"""The declarative preference schema: the latent structure v3 exists for.

A **preference dimension** is one convention the feedbacker holds — which source to trust
when they disagree, how to scope a definition, how to format a figure, what to call
things, when to escalate instead of guess. Each dimension is a data row, not a code
branch: the instance templates (``world/instances.py``) enumerate the defensible answer
under every option, the label compiler (``world/labels.py``) picks the accepted one from
a frozen profile, feedback names the violated dimension, and certification iterates the
same rows. Adding a preference is adding a row here plus a template that invokes it.

The catalog spans five kinds:

- ``source``     : which of two live-but-disagreeing surfaces is authoritative
                   (the stale ``arr`` field vs ``mrr*12``, billing vs shipping country,
                   the capped search list vs the server aggregate, refunds v1 vs v2).
- ``scope``      : how a definition is bounded (calendar vs fiscal quarters, won-only vs
                   won+open pipeline, a strict vs inclusive ageing boundary, gross
                   on-hand vs net-of-reserved inventory).
- ``format``     : how a value is presented (cents vs whole dollars, exact vs rounded to
                   the nearest thousand, status word vs raw code, ISO date vs Unix
                   seconds).
- ``naming``     : which identifier names an entity (page id vs title, account id vs
                   company name).
- ``escalation`` : when to answer at all (report an empty result vs escalate).

Conditional dimensions carry a ``condition``: the facet that gates them. ``count_authority``
and ``annual_revenue_basis`` are the same *skill* (trust which source?) gated on different
metrics, so one revealed preference does not transfer to the sibling context — that
conditionality, plus composition (instances invoke up to three dimensions at once), is
what keeps the sequence from collapsing to one step.

Every option of every dimension yields an answer **defensible from the records**: the
options are readings of real, discoverable surfaces (or conventions teachable as a class,
like the fiscal calendar), never fabrications. The reward's defensibility floor binds to
that enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("source", "scope", "format", "naming", "escalation")

# The minimum profile size the no-collapse gate enforces (design: 15-20 dimensions; a
# profile below this is one-shot-learnable and the curve degenerates to a step).
NO_COLLAPSE_MIN = 12


@dataclass(frozen=True)
class PreferenceDimension:
    """One convention the feedbacker holds, as data.

    ``id`` is the stable key profiles/labels/feedback/certification share. ``options`` is
    the closed set of defensible readings; a frozen profile commits to exactly one.
    ``applies_when`` documents which question family invokes the dimension (the tag
    templates set in ``Instance.invoked``). ``condition`` is set only on *conditional*
    dimensions: the facet gating them, so the same skill under a different facet is a
    different dimension.
    """

    id: str
    kind: str
    options: tuple[str, ...]
    applies_when: str
    condition: str | None = None


SCHEMA: tuple[PreferenceDimension, ...] = (
    # ------------------------------------------------------------------ source
    PreferenceDimension(
        id="annual_revenue_basis",
        kind="source",
        options=("arr_field", "mrr_derived"),
        applies_when="annual-revenue-for-account questions",
        condition="metric=annual_revenue",
    ),
    PreferenceDimension(
        id="country_basis",
        kind="source",
        options=("billing", "shipping"),
        applies_when="which-country-is-an-account-in questions",
    ),
    PreferenceDimension(
        id="count_authority",
        kind="source",
        options=("search", "aggregate"),
        applies_when="how-many-accounts questions where the search list and the server aggregate disagree",
        condition="metric=count",
    ),
    PreferenceDimension(
        id="process_page_authority",
        kind="source",
        options=("v1", "v2"),
        applies_when=(
            "which-page-governs-a-process questions, for any governed original/migrated "
            "page pair (both pages claim currency)"
        ),
    ),
    # ------------------------------------------------------------------- scope
    PreferenceDimension(
        id="quarter_calendar",
        kind="scope",
        options=("calendar", "fiscal"),
        applies_when="quarter-scoped revenue questions",
    ),
    PreferenceDimension(
        id="revenue_stage_scope",
        kind="scope",
        options=("won_only", "won_plus_open"),
        applies_when="booked-revenue questions that do not pin the stage",
    ),
    PreferenceDimension(
        id="overdue_boundary",
        kind="scope",
        options=("strict", "inclusive"),
        applies_when="over-N-days ageing questions (is exactly N days in or out)",
    ),
    PreferenceDimension(
        id="inventory_scope",
        kind="scope",
        options=("on_hand", "net_of_reserved"),
        applies_when="units-available inventory questions",
    ),
    # ------------------------------------------------------------------ format
    PreferenceDimension(
        id="money_unit",
        kind="format",
        options=("cents", "dollars"),
        applies_when="every monetary answer",
    ),
    PreferenceDimension(
        id="money_rounding",
        kind="format",
        options=("exact", "thousands"),
        applies_when="monetary aggregate totals",
        condition="aggregate totals only (single-record figures are always exact)",
    ),
    PreferenceDimension(
        id="status_style",
        kind="format",
        options=("word", "code"),
        applies_when="order-status questions",
    ),
    PreferenceDimension(
        id="date_style",
        kind="format",
        options=("iso", "unix"),
        applies_when="date answers",
    ),
    # ------------------------------------------------------------------ naming
    PreferenceDimension(
        id="page_naming",
        kind="naming",
        options=("id", "title"),
        applies_when="answers naming an SOP page",
    ),
    PreferenceDimension(
        id="account_naming",
        kind="naming",
        options=("id", "name"),
        applies_when="answers naming an account",
    ),
    # -------------------------------------------------------------- escalation
    PreferenceDimension(
        id="missing_data_policy",
        kind="escalation",
        options=("report_zero", "escalate"),
        applies_when="count questions whose true result is empty",
    ),
)

_BY_ID = {d.id: d for d in SCHEMA}


def dimension(dim_id: str) -> PreferenceDimension:
    """The schema row for ``dim_id`` (KeyError on an unknown id)."""
    return _BY_ID[dim_id]


def dimension_ids() -> tuple[str, ...]:
    return tuple(d.id for d in SCHEMA)


def conditional_dimensions() -> tuple[PreferenceDimension, ...]:
    return tuple(d for d in SCHEMA if d.condition is not None)


def validate_schema(schema: tuple[PreferenceDimension, ...] = SCHEMA) -> None:
    """Structural well-formedness. Raises ``ValueError`` on the first violation.

    Guards: catalog size clears the no-collapse floor, ids unique, kinds valid and all
    present, option sets non-trivial (>= 2 distinct options), conditional dimensions
    exist (>= 2) and carry a non-empty condition. ``schema`` defaults to the shipped
    catalog; tests and certification pass broken variants as positive controls.
    """
    if len(schema) < NO_COLLAPSE_MIN:
        raise ValueError(f"schema has {len(schema)} dimensions < NO_COLLAPSE_MIN {NO_COLLAPSE_MIN}")
    ids = [d.id for d in schema]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate dimension ids in SCHEMA")
    for d in schema:
        if d.kind not in KINDS:
            raise ValueError(f"dimension {d.id!r} has invalid kind {d.kind!r}")
        if len(d.options) < 2 or len(set(d.options)) != len(d.options):
            raise ValueError(f"dimension {d.id!r} needs >= 2 distinct options, got {d.options}")
        if not d.applies_when.strip():
            raise ValueError(f"dimension {d.id!r} has an empty applies_when")
        if d.condition is not None and not d.condition.strip():
            raise ValueError(f"dimension {d.id!r} has an empty condition string")
    missing_kinds = set(KINDS) - {d.kind for d in schema}
    if missing_kinds:
        raise ValueError(f"schema is missing kinds: {sorted(missing_kinds)}")
    conditional = tuple(d for d in schema if d.condition is not None)
    if len(conditional) < 2:
        raise ValueError("schema needs >= 2 conditional dimensions (the anti-transfer lever)")
