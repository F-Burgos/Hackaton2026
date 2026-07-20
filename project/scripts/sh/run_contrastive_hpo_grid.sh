#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

SWEEP_DIR="${SWEEP_DIR:-project/results/contrastive/hpo_loss_lr_temp}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-16384}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-6}"
MIN_DELTA="${MIN_DELTA:-0.002}"
MIN_LR="${MIN_LR:-0.00001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

CONFIGS=(
  "lr1e4 0.0001 temp010 0.10 symmetric_info_nce sym"
  "lr1e4 0.0001 temp010 0.10 image_to_spectrum_info_nce i2s"
  "lr1e4 0.0001 temp010 0.10 spectrum_to_image_info_nce s2i"
  "lr1e4 0.0001 temp015 0.15 symmetric_info_nce sym"
  "lr1e4 0.0001 temp015 0.15 image_to_spectrum_info_nce i2s"
  "lr1e4 0.0001 temp015 0.15 spectrum_to_image_info_nce s2i"
  "lr3e4 0.0003 temp010 0.10 symmetric_info_nce sym"
  "lr3e4 0.0003 temp010 0.10 image_to_spectrum_info_nce i2s"
  "lr3e4 0.0003 temp010 0.10 spectrum_to_image_info_nce s2i"
  "lr3e4 0.0003 temp015 0.15 symmetric_info_nce sym"
  "lr3e4 0.0003 temp015 0.15 image_to_spectrum_info_nce i2s"
  "lr3e4 0.0003 temp015 0.15 spectrum_to_image_info_nce s2i"
  "lr1e3 0.001 temp010 0.10 symmetric_info_nce sym"
  "lr1e3 0.001 temp010 0.10 image_to_spectrum_info_nce i2s"
  "lr1e3 0.001 temp010 0.10 spectrum_to_image_info_nce s2i"
  "lr1e3 0.001 temp015 0.15 symmetric_info_nce sym"
  "lr1e3 0.001 temp015 0.15 image_to_spectrum_info_nce i2s"
  "lr1e3 0.001 temp015 0.15 spectrum_to_image_info_nce s2i"
)

mkdir -p "${SWEEP_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
  read -r LR_TAG LR TEMP_TAG TEMP LOSS LOSS_TAG <<< "${CONFIG}"
  RUN_NAME="${LR_TAG}_${TEMP_TAG}_${LOSS_TAG}"
  RUN_DIR="${SWEEP_DIR}/${RUN_NAME}"
  mkdir -p "${RUN_DIR}"

  if [[ -f "${RUN_DIR}/summary.json" ]]; then
    printf '{"event":"skip_existing","run":"%s","timestamp":"%s"}\n' \
      "${RUN_NAME}" "$(date --iso-8601=seconds)" | tee -a "${SWEEP_DIR}/sweep_events.jsonl"
    continue
  fi

  printf '{"event":"start","run":"%s","learning_rate":%s,"temperature":%s,"loss":"%s","timestamp":"%s"}\n' \
    "${RUN_NAME}" "${LR}" "${TEMP}" "${LOSS}" "$(date --iso-8601=seconds)" \
    | tee -a "${SWEEP_DIR}/sweep_events.jsonl"

  bash project/scripts/sh/run_contrastive.sh \
    train.epochs="${EPOCHS}" \
    train.max_train_samples="${MAX_TRAIN_SAMPLES}" \
    train.max_val_samples="${MAX_VAL_SAMPLES}" \
    data.batch_size="${BATCH_SIZE}" \
    train.contrastive_accumulation_steps="${ACCUM_STEPS}" \
    train.contrastive_loss="${LOSS}" \
    train.temperature="${TEMP}" \
    train.learning_rate="${LR}" \
    train.gradient_clip_norm="${GRAD_CLIP}" \
    train.lr_scheduler=cosine \
    train.min_learning_rate="${MIN_LR}" \
    train.early_stopping_patience="${PATIENCE}" \
    train.early_stopping_min_delta="${MIN_DELTA}" \
    outputs.contrastive_dir="${RUN_DIR}" > "${RUN_DIR}/train.log" 2>&1

  bash project/scripts/sh/export_contrastive_embeddings.sh \
    eval.checkpoint_path="${RUN_DIR}/best.pt" \
    eval.split=val \
    eval.max_samples="${MAX_VAL_SAMPLES}" \
    eval.output_dir="${RUN_DIR}/export_val${MAX_VAL_SAMPLES}" > "${RUN_DIR}/export_val.log" 2>&1

  bash project/scripts/sh/export_contrastive_embeddings.sh \
    eval.checkpoint_path="${RUN_DIR}/best.pt" \
    eval.split=test \
    eval.max_samples="${MAX_VAL_SAMPLES}" \
    eval.output_dir="${RUN_DIR}/export_test${MAX_VAL_SAMPLES}" > "${RUN_DIR}/export_test.log" 2>&1

  bash project/scripts/sh/report_contrastive_run.sh \
    --run-dir "${RUN_DIR}" \
    --export-dir "${RUN_DIR}/export_val${MAX_VAL_SAMPLES}" \
    --output-path "${RUN_DIR}/report_val${MAX_VAL_SAMPLES}.md" > "${RUN_DIR}/report_val.log" 2>&1

  bash project/scripts/sh/report_contrastive_run.sh \
    --run-dir "${RUN_DIR}" \
    --export-dir "${RUN_DIR}/export_test${MAX_VAL_SAMPLES}" \
    --output-path "${RUN_DIR}/report_test${MAX_VAL_SAMPLES}.md" > "${RUN_DIR}/report_test.log" 2>&1

  printf '{"event":"done","run":"%s","learning_rate":%s,"temperature":%s,"loss":"%s","timestamp":"%s"}\n' \
    "${RUN_NAME}" "${LR}" "${TEMP}" "${LOSS}" "$(date --iso-8601=seconds)" \
    | tee -a "${SWEEP_DIR}/sweep_events.jsonl"
done
