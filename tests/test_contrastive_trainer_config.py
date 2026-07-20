from __future__ import annotations

import pytest

pytest.importorskip("torch")

from project.src.training.contrastive_trainer import (  # noqa: E402
    ContrastiveTrainConfig,
    _validate_config,
)


def test_contrastive_accumulation_steps_must_be_positive() -> None:
    config = ContrastiveTrainConfig(contrastive_accumulation_steps=0)

    with pytest.raises(ValueError, match="contrastive_accumulation_steps"):
        _validate_config(config)


def test_contrastive_loss_name_must_be_supported() -> None:
    config = ContrastiveTrainConfig(contrastive_loss="not_a_loss")

    with pytest.raises(ValueError, match="contrastive_loss"):
        _validate_config(config)


def test_trainable_temperature_initial_value_must_be_inside_bounds() -> None:
    config = ContrastiveTrainConfig(
        temperature=0.005,
        temperature_trainable=True,
        temperature_min=0.01,
        temperature_max=1.0,
    )

    with pytest.raises(ValueError, match="initial temperature"):
        _validate_config(config)
