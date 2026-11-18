# Program: karl-offline-lab - test_auto_train_lora_config.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for LoRA training config helpers: target module detection and compatibility filtering of config arguments.
"""Tests for tools.auto_train_lora config helpers."""

import argparse
import importlib.util
from pathlib import Path

import pytest

from tools.auto_train_lora import (
    build_sft_config_kwargs,
    detect_target_modules,
    filter_kwargs_for_class,
)


class FakeModel:
    """Minimal model stand-in exposing a list of named modules."""
    def named_modules(self):
        """Return (name, module) pairs for typical transformer layer names."""
        # Typical layer names found in a transformer, plus the output head
        names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
            "lm_head",
        ]
        return [(name, object()) for name in names]


def test_detect_target_modules_keeps_known_projection_order():
    """Scenario: a model with attention and MLP projections. Expected: target modules come back in the canonical order without the output head."""
    assert detect_target_modules(FakeModel()) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_detect_target_modules_has_attention_fallback():
    """Scenario: a model with no known projections. Expected: falls back to q_proj and v_proj."""
    class EmptyModel:
        """Model stand-in with no projection layers."""
        def named_modules(self):
            """Return a single non-projection module."""
            return [("embed_tokens", object())]

    assert detect_target_modules(EmptyModel()) == ["q_proj", "v_proj"]


def test_build_sft_config_enables_packing_and_cosine_schedule():
    """Scenario: build config kwargs from typical args. Expected: packing on, cosine schedule, warmup ratio and text field set."""
    args = argparse.Namespace(
        max_length=4096,
        batch_size=1,
        grad_accum=8,
        lr=2e-4,
        warmup_ratio=0.03,
        epochs=1,
        logging_steps=1,
    )

    kwargs = build_sft_config_kwargs(args, Path("data/adapters/test/checkpoints"))

    assert kwargs["packing"] is True
    assert kwargs["max_length"] == 4096
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["warmup_ratio"] == 0.03
    assert kwargs["dataset_text_field"] == "text"


def test_sft_config_accepts_packing_and_cosine_kwargs_when_trl_available():
    """SFTConfig construction must survive whichever trl version is actually
    installed -- requirements.txt doesn't pin trl, and SFTConfig's accepted
    fields (e.g. warmup_ratio) have moved between releases before. Goes
    through the same filter_kwargs_for_class() compatibility layer train()
    uses, rather than asserting the raw kwargs dict always matches whatever
    SFTConfig happens to accept in the installed version."""
    if importlib.util.find_spec("trl") is None:
        pytest.skip("trl not installed")
    from trl import SFTConfig

    args = argparse.Namespace(
        max_length=4096,
        batch_size=1,
        grad_accum=8,
        lr=2e-4,
        warmup_ratio=0.03,
        epochs=1,
        logging_steps=1,
    )
    # Build raw kwargs, then filter them exactly as the trainer does
    raw_kwargs = build_sft_config_kwargs(args, Path("data/adapters/test/checkpoints"))
    cfg = SFTConfig(**filter_kwargs_for_class(raw_kwargs, SFTConfig))

    assert cfg.packing is True
    assert cfg.max_length == 4096
    assert str(cfg.lr_scheduler_type) == "SchedulerType.COSINE"


@pytest.mark.parametrize("value", [True, False])
def test_checkpointing_uses_legacy_name_on_older_config(value):
    """Scenario: config class only knows the legacy checkpointing names. Expected: values pass through unchanged and the input dict is not mutated."""
    class OldConfig:
        """Config stand-in using the legacy gradient checkpointing argument names."""
        def __init__(self, gradient_checkpointing=False, gradient_checkpointing_kwargs=None):
            """Store the checkpointing flag and options."""
            self.enabled = gradient_checkpointing
            self.options = gradient_checkpointing_kwargs

    options = {"use_reentrant": False}
    raw = {"gradient_checkpointing": value, "gradient_checkpointing_kwargs": options}
    cfg = OldConfig(**filter_kwargs_for_class(raw, OldConfig))
    assert cfg.enabled is value
    assert cfg.options == options
    assert raw == {"gradient_checkpointing": value, "gradient_checkpointing_kwargs": options}


@pytest.mark.parametrize("value", [True, False])
def test_checkpointing_prefers_new_name_without_deprecated_alias(value):
    """Scenario: config class knows the new activation checkpointing names. Expected: legacy names are translated and never passed through."""
    class NewConfig:
        """Config stand-in that rejects the deprecated names."""
        def __init__(self, activation_checkpointing=False, activation_checkpointing_kwargs=None,
                     gradient_checkpointing=None, gradient_checkpointing_kwargs=None):
            """Raise if a deprecated argument arrives, otherwise store the new ones."""
            if gradient_checkpointing is not None or gradient_checkpointing_kwargs is not None:
                raise AssertionError("deprecated checkpointing arguments passed")
            self.enabled = activation_checkpointing
            self.options = activation_checkpointing_kwargs

    options = {"use_reentrant": False}
    raw = {"gradient_checkpointing": value, "gradient_checkpointing_kwargs": options}
    cfg = NewConfig(**filter_kwargs_for_class(raw, NewConfig))
    assert cfg.enabled is value
    assert cfg.options == options


def test_explicit_new_checkpointing_setting_takes_precedence():
    """Scenario: both legacy and new names are supplied. Expected: the explicit new setting wins."""
    class NewConfig:
        """Config stand-in accepting both naming schemes."""
        def __init__(self, activation_checkpointing=None, gradient_checkpointing=None):
            """Accept the arguments and ignore them."""
            pass

    raw = {"gradient_checkpointing": True, "activation_checkpointing": False}
    assert filter_kwargs_for_class(raw, NewConfig) == {"activation_checkpointing": False}


def test_checkpointing_unknown_signature_keeps_legacy_and_other_kwargs():
    """Scenario: config accepts arbitrary keyword arguments. Expected: all kwargs, legacy included, are kept."""
    class FlexibleConfig:
        """Config stand-in that accepts any keyword arguments."""
        def __init__(self, **kwargs):
            """Keep every keyword argument."""
            self.kwargs = kwargs

    raw = {"gradient_checkpointing": True, "future_option": "x"}
    assert FlexibleConfig(**filter_kwargs_for_class(raw, FlexibleConfig)).kwargs == raw


def test_checkpointing_without_supported_parameter_reports_drop(capsys):
    """Scenario: config has no checkpointing parameter. Expected: the option is dropped and a note naming it is written to stderr."""
    class NoCheckpointConfig:
        """Config stand-in with only an output directory."""
        def __init__(self, output_dir):
            """Accept the output directory only."""
            pass

    assert filter_kwargs_for_class({"output_dir": "out", "gradient_checkpointing": True}, NoCheckpointConfig) == {"output_dir": "out"}
    assert "gradient_checkpointing" in capsys.readouterr().err


def test_checkpointing_accepts_new_only_signature():
    """Scenario: config only knows the new name. Expected: the legacy flag is translated and applied."""
    class NewOnlyConfig:
        """Config stand-in that only knows the new name."""
        def __init__(self, activation_checkpointing=False):
            """Store the new checkpointing flag."""
            self.enabled = activation_checkpointing

    cfg = NewOnlyConfig(**filter_kwargs_for_class({"gradient_checkpointing": True}, NewOnlyConfig))
    assert cfg.enabled is True


def test_checkpointing_adapts_explicit_new_name_with_flexible_kwargs():
    """Scenario: config knows the new name plus arbitrary kwargs. Expected: the flag is translated and other kwargs are preserved."""
    class NewFlexibleConfig:
        """Config stand-in with the new name and extra keyword arguments."""
        def __init__(self, activation_checkpointing=False, **kwargs):
            """Store the flag and the extra keyword arguments."""
            self.enabled = activation_checkpointing
            self.kwargs = kwargs

    cfg = NewFlexibleConfig(**filter_kwargs_for_class({"gradient_checkpointing": True, "other": 2}, NewFlexibleConfig))
    assert cfg.enabled is True
    assert cfg.kwargs == {"other": 2}
