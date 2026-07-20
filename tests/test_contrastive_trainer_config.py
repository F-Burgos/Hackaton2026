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
