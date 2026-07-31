"""Frozen dataclasses for the hydrated world (v4: loaded from stored data, never
generated at runtime).

Records use string FK keys (not nested objects) so referential integrity is checkable
and equality is cheap. Record stores are `Mapping[str, Record]` keyed by id; the store
wraps them in `MappingProxyType` so the world is genuinely immutable and
equality-comparable. The `Behavior` block carries the resolved API/convention behaviors
the tools realize (what v3 called quirk params, now plain committed data).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# CRM/ERP records. All monetary amounts are integer cents; all timestamps are   #
# Unix seconds.                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Account:
    id: str  # "acc-00042"
    name: str
    region: int  # 0..R-1 ; the region-filter quirk (02-03) bites here
    segment: str  # "enterprise" | "midmarket" | "smb"
    mrr: int  # actual monthly recurring revenue (cents)
    arr: int  # documented annual figure; quirk 7 makes this stale vs mrr
    billing_country: str  # ISO-3166 alpha-2
    shipping_country: str  # ISO-3166 alpha-2
    created_at: int  # Unix seconds
    churned: bool
    churn_campaign: str | None  # campaign label if churned, else None


@dataclass(frozen=True)
class Contact:
    id: str
    account_id: str  # FK -> Account
    name: str
    email: str
    role: str


@dataclass(frozen=True)
class Opportunity:
    id: str  # "opp-00123" (raw; quirk 2 prefixes it at lookup)
    account_id: str  # FK -> Account
    name: str
    amount: int  # booked value (cents)
    stage: str  # "open" | "won" | "lost"
    close_date: int  # Unix seconds; quirk 3 reinterprets the quarter mapping
    region: int  # denormalized region for revenue-by-region questions


@dataclass(frozen=True)
class Order:
    id: str
    account_id: str  # FK -> Account
    product_id: str  # FK -> Product (primary line item)
    quantity: int
    amount: int  # cents
    status_code: int  # opaque integer status (quirk 5 maps code -> meaning)
    created_at: int  # Unix seconds


@dataclass(frozen=True)
class Invoice:
    id: str
    order_id: str  # FK -> Order
    account_id: str  # FK -> Account (denormalized)
    region: int  # denormalized region (the region-filter quirk bites here)
    amount: int  # cents
    paid: bool
    issued_at: int  # Unix seconds
    days_outstanding: int  # age used by the "over 60 days" question


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    unit_price: int  # cents
    category: str


@dataclass(frozen=True)
class InventoryItem:
    id: str
    product_id: str  # FK -> Product
    warehouse: str
    on_hand: int
    reserved: int


@dataclass(frozen=True)
class Ticket:
    id: str
    account_id: str  # FK -> Account
    subject: str
    priority: str
    status: str
    opened_at: int  # Unix seconds


# --------------------------------------------------------------------------- #
# SOP / Notion knowledge base (02-02).                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SopPage:
    id: str  # "sop-revenue-recognition"
    space: str  # "Finance" | "Sales" | "Ops" | "Support"
    title: str
    body: str  # 1k-2k tokens of prose


@dataclass(frozen=True)
class ChangelogEntry:
    date: str  # ISO date
    author: str
    note: str


@dataclass(frozen=True)
class SopKnowledgeBase:
    pages: Mapping[str, SopPage]
    glossary: Mapping[str, str]  # field name -> documented meaning
    runbooks: Mapping[str, SopPage]
    changelog: tuple[ChangelogEntry, ...]


# --------------------------------------------------------------------------- #
# Behavior: the world's resolved API/convention behaviors, hydrated from the    #
# stored world file (what v3 called quirk params, now plain committed data).    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GovernedPair:
    """An original/migrated process-page pair where both pages claim currency."""

    topic: str  # "refund approvals"
    original: str  # "sop-refunds-v1"
    migrated: str  # "sop-refunds-v2"


@dataclass(frozen=True)
class Behavior:
    search_cap: int  # search_accounts silently caps at this many matches
    opportunity_lookup_prefix: str  # get_opportunity needs the prefixed key
    fiscal_year_start_month: int  # the company fiscal calendar (2..12)
    report_stale_on_retry: bool  # a repeated query_report call silently goes stale
    region_filter_exclusive_invoices: bool  # region filter is EXCLUSIVE on invoices
    deprecated_inventory_endpoint: str  # documented route that 404s
    working_inventory_endpoint: str  # undocumented route that serves
    status_code_table: tuple[str, ...]  # order status_code -> word
    governed_page_pairs: tuple[GovernedPair, ...]


# --------------------------------------------------------------------------- #
# The hydrated world.                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class World:
    """A fully populated, frozen world hydrated from its stored file:
    referentially-consistent CRM/ERP records, the SOP/Notion knowledge base, and the
    behavior block. Same file -> identical world (the determinism contract)."""

    name: str
    accounts: Mapping[str, Account]
    contacts: Mapping[str, Contact]
    opportunities: Mapping[str, Opportunity]
    orders: Mapping[str, Order]
    invoices: Mapping[str, Invoice]
    products: Mapping[str, Product]
    inventory: Mapping[str, InventoryItem]
    tickets: Mapping[str, Ticket]
    sop: SopKnowledgeBase
    behavior: Behavior
