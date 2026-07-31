"""The alien-api world layer (v4: hydration only, no generation).

The environment never generates a world: it hydrates the ONE committed world from its
stored file (``store.load_world``) and serves it through the tools. What lives here:

- ``models``      : the frozen dataclass tree, including the ``Behavior`` block (the
                    resolved API/convention behaviors the tools realize);
- ``store``       : world.json -> ``World``, cached per process;
- ``conventions`` : the shared deterministic seam (fiscal quarters, the lookup key, the
                    money/date renderers) tools and the build-time templates both use;
- ``preferences`` : the declarative schema of Margot's 15 preference dimensions
                    (feedback and violation attribution read it);
- ``profile``     : her frozen profile (labels were compiled against it at build time);
- ``value_extract``: the weight-0 ``value_correct`` diagnostic matcher.

Generation (records, SOP, quirks, instances, labels) lives in ``alien-api-synth``.
"""

from __future__ import annotations

from alien_api_env.world import (
    conventions,
    preferences,
    profile,
    store,
    value_extract,
)
from alien_api_env.world.models import (
    Account,
    Behavior,
    ChangelogEntry,
    Contact,
    GovernedPair,
    InventoryItem,
    Invoice,
    Opportunity,
    Order,
    Product,
    SopKnowledgeBase,
    SopPage,
    Ticket,
    World,
)

__all__ = [
    "World",
    "Behavior",
    "GovernedPair",
    "conventions",
    "preferences",
    "profile",
    "store",
    "value_extract",
    "Account",
    "Contact",
    "Opportunity",
    "Order",
    "Invoice",
    "Product",
    "InventoryItem",
    "Ticket",
    "SopPage",
    "SopKnowledgeBase",
    "ChangelogEntry",
]
