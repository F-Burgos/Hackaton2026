from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from project.src.downstream.anomalies import apply_anomalies_to_batch, build_anomaly_manifest


def test_anomaly_manifest_is_deterministic_and_balanced() -> None:
    object_ids = tuple(f"obj_{idx}" for idx in range(20))

    first = build_anomaly_manifest(object_ids, fraction=0.10, seed=7, modality="alternate")
    second = build_anomaly_manifest(object_ids, fraction=0.10, seed=7, modality="alternate")

    assert first == second
    assert len(first) == 2
    assert {spec.modality for spec in first.values()} == {"image", "spectrum"}


def test_apply_anomalies_changes_only_selected_objects() -> None:
    object_ids = ["normal", "anomaly"]
    manifest = build_anomaly_manifest(
        object_ids,
        fraction=0.50,
        seed=3,
        modality="image",
        image_kind="bright_patch",
        strength=4.0,
    )
    selected = next(iter(manifest))
    batch = {
        "object_id": object_ids,
        "img": torch.zeros(2, 9, 64, 64),
        "img_channel_mask": torch.ones(2, 9),
        "flux_lambda_normalized": torch.zeros(2, 739),
        "mask_spectra": torch.ones(2, 739),
        "wave": torch.zeros(2, 739),
    }

    output = apply_anomalies_to_batch(batch, manifest)

    changed = output["img"].abs().sum(dim=(1, 2, 3)) > 0
    expected = torch.tensor([object_id == selected for object_id in object_ids])
    assert torch.equal(changed.cpu(), expected)
    assert torch.equal(output["flux_lambda_normalized"], batch["flux_lambda_normalized"])
