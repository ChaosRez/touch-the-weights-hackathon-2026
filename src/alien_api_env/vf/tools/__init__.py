"""The alien-api tool surface (verifiers-v1 MCP toolsets).

Two stateless toolsets served over MCP: ``CrmToolset`` (CRM/ERP records, realizing the
behavioral quirks + authoritative reference endpoints) and ``WikiToolset`` (the drifting
SOP/wiki). Both build the Phase 2 world once per task in ``setup_task``. No scratchpad /
notepad / memory tool — memory is a property of the system under test, excluded by design.

``ARTIFACT_VERBOSITY`` is the per-instance artifact-token budget list returns are padded
toward (``envelopes.pad_to_tokens``); ``count_tokens`` sizes the payloads and the weight-0
``artifact_tokens`` metric.
"""

from alien_api_env.vf.tools.answer import (
    SUBMIT_ANSWER_TOOL,
    AnswerToolset,
    is_submit_answer,
    submitted_value,
)
from alien_api_env.vf.tools.crm import CrmToolset
from alien_api_env.vf.tools.envelopes import (
    ARTIFACT_VERBOSITY,
    count_tokens,
    pad_to_tokens,
)
from alien_api_env.vf.tools.wiki import WikiToolset

__all__ = [
    "CrmToolset",
    "WikiToolset",
    "AnswerToolset",
    "SUBMIT_ANSWER_TOOL",
    "submitted_value",
    "is_submit_answer",
    "ARTIFACT_VERBOSITY",
    "count_tokens",
    "pad_to_tokens",
]
