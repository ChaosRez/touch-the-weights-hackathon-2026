"""World hydration: the stored world file -> the frozen `World` object.

The environment's only source of world data. Reads `worlds/<name>/world.json` from the
packaged data directory (or an explicit `worlds_root`), validates the minimal shape, and
constructs the immutable dataclass tree the tools serve. Cached per (name, root) and
process — the same file always hydrates the identical world, and toolset processes pay
the parse once.

No generation, no seeds, no model: if the file is missing or malformed this raises
loudly. Certification of the *content* happened at build time in `alien-api-synth`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from alien_api_env.world import models as M

# The packaged fleet (shipped in the wheel). An explicit worlds_root overrides it.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DEFAULT_WORLD = "kestrel"

# The data stores (git holds no data). The GCS artifact store is first-class (the
# milestone pattern: immutable digest-verified versions, offline once warm); the HF
# dataset is the open-source mirror of the same audited bytes. Both are written only
# by alien-api-synth's publish scripts, after the audit.
DEFAULT_ARTIFACT = "alien-api/v4.1.0"  # name/version in the artifact store
DEFAULT_HF_DATASET = "hf://ConstructLabs/alien-api"
HF_PREFIX = "hf://"


@lru_cache(maxsize=4)
def pull_artifact(ref: str = DEFAULT_ARTIFACT, remote: str = ""):
    """Materialize an artifact-store version ("name/version") into the digest-verified
    cache; returns the pull dict ({'files', 'blobs'} of local paths). Lazy import — the
    optional ``gcs`` extra; remote defaults to ``CS_ARTIFACTS_REMOTE``."""
    from artifact_store import store as artifacts

    name, _, version = ref.partition("/")
    if not version:
        raise ValueError(f"artifact must be 'name/version' (got {ref!r})")
    return artifacts.pull(remote or artifacts.default_remote(), name, version)


def resolve_hf(ref: str, filename: str) -> Path:
    """``hf://org/name`` + a repo-relative filename -> a locally cached file.

    Uses the huggingface_hub cache (token from ``HF_TOKEN`` for private repos). The
    dependency is the optional ``hf`` extra; the bundled data needs none of this.
    """
    from huggingface_hub import hf_hub_download  # lazy: optional extra

    repo_id = ref.removeprefix(HF_PREFIX).strip("/")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))

_STORES: dict[str, type] = {
    "accounts": M.Account,
    "contacts": M.Contact,
    "opportunities": M.Opportunity,
    "orders": M.Order,
    "invoices": M.Invoice,
    "products": M.Product,
    "inventory": M.InventoryItem,
    "tickets": M.Ticket,
}


def world_path(name: str = DEFAULT_WORLD, worlds_root: str = "") -> Path:
    if worlds_root.startswith(HF_PREFIX):
        return resolve_hf(worlds_root, f"worlds/{name}/world.json")
    if worlds_root:
        return Path(worlds_root) / name / "world.json"
    local = DATA_DIR / "worlds" / name / "world.json"
    if local.exists():
        return local  # locally installed (build_fleet.py --install); the offline path
    try:  # first-class remote: the GCS artifact store
        return Path(pull_artifact()["blobs"]["worlds"]) / name / "world.json"
    except Exception:
        pass  # no artifact-store package / creds / version — fall through to the mirror
    return resolve_hf(DEFAULT_HF_DATASET, f"worlds/{name}/world.json")


def world_from_dict(artifact: dict) -> M.World:
    """Construct the frozen `World` from a stored world artifact dict."""
    b = artifact["behavior"]
    behavior = M.Behavior(
        search_cap=int(b["search_cap"]),
        opportunity_lookup_prefix=str(b["opportunity_lookup_prefix"]),
        fiscal_year_start_month=int(b["fiscal_year_start_month"]),
        report_stale_on_retry=bool(b["report_stale_on_retry"]),
        region_filter_exclusive_invoices=bool(b["region_filter_exclusive_invoices"]),
        deprecated_inventory_endpoint=str(b["deprecated_inventory_endpoint"]),
        working_inventory_endpoint=str(b["working_inventory_endpoint"]),
        status_code_table=tuple(b["status_code_table"]),
        governed_page_pairs=tuple(M.GovernedPair(**p) for p in b["governed_page_pairs"]),
    )
    stores = {
        store: MappingProxyType({row["id"]: cls(**row) for row in artifact["records"][store]})
        for store, cls in _STORES.items()
    }
    sop = M.SopKnowledgeBase(
        pages=MappingProxyType({pid: M.SopPage(**p) for pid, p in artifact["sop"]["pages"].items()}),
        glossary=MappingProxyType(dict(artifact["sop"]["glossary"])),
        runbooks=MappingProxyType(
            {pid: M.SopPage(**p) for pid, p in artifact["sop"]["runbooks"].items()}
        ),
        changelog=tuple(M.ChangelogEntry(**e) for e in artifact["sop"]["changelog"]),
    )
    return M.World(name=artifact["name"], **stores, sop=sop, behavior=behavior)


@lru_cache(maxsize=4)
def load_world(name: str = DEFAULT_WORLD, worlds_root: str = "") -> M.World:
    """Hydrate a world by name. Cached; raises FileNotFoundError/KeyError loudly."""
    try:
        path = world_path(name, worlds_root)
    except Exception as e:  # hub resolution failed (offline / no HF_TOKEN)
        raise FileNotFoundError(
            f"no stored world {name!r}: not installed locally, artifact store "
            f"{DEFAULT_ARTIFACT!r} unavailable, and {DEFAULT_HF_DATASET} unreachable — "
            "run alien-api-synth's build_fleet.py --install, configure GCS access "
            f"(CS_ARTIFACTS_REMOTE), or set HF_TOKEN ({e})"
        ) from e
    if not path.exists():
        raise FileNotFoundError(f"no stored world {name!r} at {path}")
    return world_from_dict(json.loads(path.read_text(encoding="utf-8")))
