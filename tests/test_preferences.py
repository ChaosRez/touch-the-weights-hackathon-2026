"""Schema well-formedness (Phase 1): the declarative catalog is structurally sound and
the validator has teeth (broken variants are rejected)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from alien_api_env.world.preferences import (
    KINDS,
    NO_COLLAPSE_MIN,
    SCHEMA,
    PreferenceDimension,
    conditional_dimensions,
    dimension,
    dimension_ids,
    validate_schema,
)


def test_schema_validates():
    validate_schema()


def test_schema_size_clears_no_collapse_floor():
    assert len(SCHEMA) >= NO_COLLAPSE_MIN
    assert 15 <= len(SCHEMA) <= 20  # the design's target band


def test_ids_unique_and_lookup_works():
    ids = dimension_ids()
    assert len(set(ids)) == len(ids)
    for did in ids:
        assert dimension(did).id == did
    with pytest.raises(KeyError):
        dimension("no-such-dimension")


def test_every_kind_present_and_valid():
    kinds = {d.kind for d in SCHEMA}
    assert kinds == set(KINDS)


def test_option_sets_non_trivial():
    for d in SCHEMA:
        assert len(d.options) >= 2
        assert len(set(d.options)) == len(d.options)


def test_conditional_dimensions_present_with_real_conditions():
    conditional = conditional_dimensions()
    assert len(conditional) >= 2
    for d in conditional:
        assert d.condition and d.condition.strip()
    # The conditional pair the design names: the same trust-which-source skill gated on
    # different metrics, so one revealed preference does not transfer to the sibling.
    by_id = {d.id: d for d in conditional}
    assert "annual_revenue_basis" in by_id
    assert "count_authority" in by_id


# ------------------------------------------------------------- positive controls


def test_validator_rejects_sub_minimum_schema():
    with pytest.raises(ValueError, match="NO_COLLAPSE_MIN"):
        validate_schema(SCHEMA[: NO_COLLAPSE_MIN - 1])


def test_validator_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        validate_schema(SCHEMA + (SCHEMA[0],))


def test_validator_rejects_invalid_kind():
    broken = SCHEMA[:-1] + (replace(SCHEMA[-1], kind="vibes"),)
    with pytest.raises(ValueError, match="invalid kind"):
        validate_schema(broken)


def test_validator_rejects_trivial_option_set():
    broken = SCHEMA[:-1] + (replace(SCHEMA[-1], options=("only",)),)
    with pytest.raises(ValueError, match="2 distinct options"):
        validate_schema(broken)


def test_validator_rejects_missing_kind_coverage():
    no_escalation = tuple(
        replace(d, kind="scope") if d.kind == "escalation" else d for d in SCHEMA
    )
    with pytest.raises(ValueError, match="missing kinds"):
        validate_schema(no_escalation)


def test_validator_rejects_conditionless_schema():
    flattened = tuple(replace(d, condition=None) for d in SCHEMA)
    with pytest.raises(ValueError, match="conditional"):
        validate_schema(flattened)


def test_validator_rejects_empty_condition_string():
    broken = tuple(
        replace(d, condition="  ") if d.id == "count_authority" else d for d in SCHEMA
    )
    with pytest.raises(ValueError, match="empty condition"):
        validate_schema(broken)


def test_dimension_rows_are_frozen_data():
    with pytest.raises(AttributeError):
        SCHEMA[0].kind = "scope"  # type: ignore[misc]
    assert isinstance(SCHEMA[0], PreferenceDimension)
