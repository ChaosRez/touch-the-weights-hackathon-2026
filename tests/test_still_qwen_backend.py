from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from still._testing import make_tiny_model
from still.config import STILLConfig
from still.model.attention import invocation_count, reset_invocation_count

from cartridge_memory.models import CompactKVAttachment
from cartridge_memory.qwen_agent import (
    HuggingFaceQwenBackend,
    QwenGenerationConfig,
    StillQwenBackend,
)


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    return make_tiny_model(str(tmp_path_factory.mktemp("phase4-tiny") / "model"), seed=0)


def _checkpoint(model_name, cfg, tiny_model_path):
    from still.model.wrapper import STILLModel

    model = STILLModel(tiny_model_path, cfg=cfg, device="cpu")
    path = model_name / "checkpoint.pt"
    torch.save(
        {
            "format": "still-recurrent-v1",
            "stage": "recurrence_aware",
            "completed_steps": 1,
            "model_name": tiny_model_path,
            "config": asdict(cfg),
            "perceiver": model.perceiver.state_dict(),
        },
        path,
    )
    return path


def _backends(tmp_path, tiny_model_path):
    config = QwenGenerationConfig(
        model_name=tiny_model_path,
        device="cpu",
        dtype="float32",
        max_new_tokens=2,
        do_sample=False,
        enable_thinking=False,
    )
    still_config = STILLConfig(
        model_name=tiny_model_path,
        num_latents=4,
        latent_dim=16,
        num_blocks=2,
        device="cpu",
    )
    checkpoint = _checkpoint(tmp_path, still_config, tiny_model_path)
    return (
        HuggingFaceQwenBackend(config),
        StillQwenBackend(config, str(checkpoint), still_config=still_config),
    )


def test_cache_none_matches_cold_live_render_and_greedy_generation(tmp_path, tiny_model_path):
    cold, still = _backends(tmp_path, tiny_model_path)
    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is two plus two?"},
    ]

    cold_turn = cold._generate_sync(messages, [], None, seed=7)
    still_turn = still._generate_sync(messages, [], None, seed=7)

    assert still_turn.input_tokens == cold_turn.input_tokens
    assert still_turn.token_ids == cold_turn.token_ids
    assert still_turn.text == cold_turn.text


def test_compact_attachment_reaches_registered_still_attention(tmp_path, tiny_model_path):
    _, backend = _backends(tmp_path, tiny_model_path)
    cache = backend.still_model.compact_tokens([1, 2, 3, 4, 5])
    attachment = CompactKVAttachment(cache=cache, source_tokens=5, latent_count=4)
    reset_invocation_count()

    turn = backend._generate_sync(
        [{"role": "user", "content": "Say hello."}],
        [],
        attachment,
        seed=3,
    )

    assert turn.output_tokens > 0
    assert invocation_count() > 0
