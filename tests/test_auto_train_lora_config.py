# ==============================================================================
# Program: test_auto_train_lora_config.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising auto train lora config functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

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
    """Model double whose named_modules() lists typical transformer layer names, so projection detection can run without torch."""
    def named_modules(self):
        """Yield (name, module) pairs including q/k/v/o and MLP projection layers."""
        # Step 1: Validate input parameters, operational preconditions, and state
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
    """Detected LoRA target modules come back in the canonical projection order, not the model's traversal order."""
    # Step 1: Validate input parameters, operational preconditions, and state
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
    """A model exposing no known projections falls back to ['q_proj', 'v_proj']."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class EmptyModel:
        """Model double with no modules."""
        def named_modules(self):
            """Return no modules."""
            # Step 1: Validate input parameters, operational preconditions, and state
            return [("embed_tokens", object())]

    assert detect_target_modules(EmptyModel()) == ["q_proj", "v_proj"]


def test_build_sft_config_enables_packing_and_cosine_schedule():
    """The SFT kwargs builder turns on packing, uses a cosine LR schedule with the warmup ratio and max_length from the arguments, and trains on a 'text' field."""
    # Step 1: Validate input parameters, operational preconditions, and state
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
    # Step 1: Validate input parameters, operational preconditions, and state
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
    raw_kwargs = build_sft_config_kwargs(args, Path("data/adapters/test/checkpoints"))
    cfg = SFTConfig(**filter_kwargs_for_class(raw_kwargs, SFTConfig))

    assert cfg.packing is True
    assert cfg.max_length == 4096
    assert str(cfg.lr_scheduler_type) == "SchedulerType.COSINE"


@pytest.mark.parametrize("value", [True, False])
def test_checkpointing_uses_legacy_name_on_older_config(value):
    """For a config class that only knows gradient_checkpointing, the legacy names are kept and the value passes through unchanged."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class OldConfig:
        """Config double with only the legacy checkpointing parameters."""
        def __init__(self, gradient_checkpointing=False, gradient_checkpointing_kwargs=None):
            """Record the legacy checkpointing arguments."""
            # Step 1: Validate input parameters, operational preconditions, and state
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
    """When the class supports activation_checkpointing, the value is renamed to it and the deprecated alias is not also passed."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class NewConfig:
        """Config double with the new activation_checkpointing parameters."""
        def __init__(self, activation_checkpointing=False, activation_checkpointing_kwargs=None,
                     gradient_checkpointing=None, gradient_checkpointing_kwargs=None):
            """Record the new-style checkpointing arguments."""
            # Step 1: Validate input parameters, operational preconditions, and state
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
    """If the caller already set activation_checkpointing, it wins over the legacy key."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class NewConfig:
        """Config double accepting both spellings."""
        def __init__(self, activation_checkpointing=None, gradient_checkpointing=None):
            """Accept either checkpointing spelling."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

    raw = {"gradient_checkpointing": True, "activation_checkpointing": False}
    assert filter_kwargs_for_class(raw, NewConfig) == {"activation_checkpointing": False}


def test_checkpointing_unknown_signature_keeps_legacy_and_other_kwargs():
    """A class taking **kwargs receives all kwargs unchanged (nothing to filter)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class FlexibleConfig:
        """Config double that accepts arbitrary keywords."""
        def __init__(self, **kwargs):
            """Store whatever kwargs it receives."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.kwargs = kwargs

    raw = {"gradient_checkpointing": True, "future_option": "x"}
    assert FlexibleConfig(**filter_kwargs_for_class(raw, FlexibleConfig)).kwargs == raw


def test_checkpointing_without_supported_parameter_reports_drop(capsys):
    """When the class has no checkpointing parameter at all, the option is dropped and a warning naming it is printed to stderr."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class NoCheckpointConfig:
        """Config double with only output_dir."""
        def __init__(self, output_dir):
            """Accept only output_dir."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

    assert filter_kwargs_for_class({"output_dir": "out", "gradient_checkpointing": True}, NoCheckpointConfig) == {"output_dir": "out"}
    assert "gradient_checkpointing" in capsys.readouterr().err


def test_checkpointing_accepts_new_only_signature():
    """A class that only knows the new parameter name still gets the requested value."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class NewOnlyConfig:
        """Config double with only the new parameter."""
        def __init__(self, activation_checkpointing=False):
            """Record activation_checkpointing."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.enabled = activation_checkpointing

    cfg = NewOnlyConfig(**filter_kwargs_for_class({"gradient_checkpointing": True}, NewOnlyConfig))
    assert cfg.enabled is True


def test_checkpointing_adapts_explicit_new_name_with_flexible_kwargs():
    """A class with the new parameter plus **kwargs gets the new name and keeps its other keywords."""
    # Step 1: Validate input parameters, operational preconditions, and state
    class NewFlexibleConfig:
        """Config double with the new parameter and **kwargs."""
        def __init__(self, activation_checkpointing=False, **kwargs):
            """Record the new parameter and the remaining kwargs."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.enabled = activation_checkpointing
            self.kwargs = kwargs

    cfg = NewFlexibleConfig(**filter_kwargs_for_class({"gradient_checkpointing": True, "other": 2}, NewFlexibleConfig))
    assert cfg.enabled is True
    assert cfg.kwargs == {"other": 2}
