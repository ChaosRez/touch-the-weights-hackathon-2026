from __future__ import annotations

import pytest

from cartridge_memory.models import CompactKVAttachment, RawKVAttachment, TextAttachment
from cartridge_memory.qwen_agent import HuggingFaceQwenBackend, UnsupportedAttachmentError


def test_text_attachment_prefixes_only_the_first_user_message() -> None:
    original = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "result"},
    ]

    attached = HuggingFaceQwenBackend._attach_text(original, TextAttachment("remember this"))

    assert attached[1]["content"] == (
        "Retrieved memory for this task:\nremember this\n\nCurrent task:\nquestion"
    )
    assert attached[3]["content"] == "result"
    assert original[1]["content"] == "question"


@pytest.mark.parametrize(
    "attachment",
    [
        RawKVAttachment(cache=object(), source_tokens=10),
        CompactKVAttachment(cache=object(), source_tokens=10, latent_count=2),
    ],
)
def test_kv_attachments_fail_loudly_until_their_phase(attachment: object) -> None:
    with pytest.raises(UnsupportedAttachmentError):
        HuggingFaceQwenBackend._attach_text([], attachment)  # type: ignore[arg-type]
