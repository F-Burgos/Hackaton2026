# Hackaton2026 Project Status

Ultima actualizacion: 2026-07-16

## Estado Actual

Estamos en fase de preparacion tecnica y entrenamiento contrastivo base. Ya existen utilidades de acceso lazy a datos, filtrado de particiones, smoke tests SSL/contrastive, entrenamiento contrastivo base, metricas retrieval, export de embeddings por split y diagnosticos ligeros del espacio latente. Todavia no hay HPO, entrenamiento largo, reportes cientificos finales ni downstream de anomalias implementado.

## Decisiones Tomadas

- El dataset pesado vive en `hackaton/` y no se versiona en Git.
- Los articulos viven en `Useful_articles/` y tampoco se versionan.
- Outputs livianos de evidencia bajo `project/results/` (`.md`, `.json`, `.jsonl`, `.csv`, `.png`) se versionan; checkpoints, embeddings `.npz`, logs crudos y `outputs/` runtime se ignoran.
- `CLAUDE.md` queda como referencia local ignorada por Git; el archivo operativo versionado es `Codex.md`.
- El desarrollo de codigo se hara localmente.
- Los dry runs se haran localmente cuando sea posible.
- Los experimentos completos se correran en un servidor remoto GPU/HPC privado, dentro de `~/Hackaton2026`.
- En el servidor remoto, el codigo debe actualizarse mediante `git pull`; no se editaran archivos directamente alla.
- El entorno comun del proyecto sigue lo que el servidor remoto puede ejecutar: Python 3.10, `uv`, `.venv` local al repo y `--system-site-packages` para reutilizar PyTorch/CUDA del sistema.
- El downstream principal sera investigacion de anomalias en el espacio latente multimodal.
- Crossmodal feature mapping queda como follow-up, no como primera version.
- El objetivo amplio post-hackaton incluye estudiar como integrar agentes y modelos de lenguaje en la investigacion cientifica, manteniendo trazabilidad, reproducibilidad y juicio cientifico humano.
- Codex debe entregar reportes de progreso frecuentes durante trabajo local, ejecuciones remotas, experimentos y cambios de direccion.

## Artefactos Creados

- `Codex.md`: instrucciones de workflow del proyecto.
- `data/reports/dataset_structure_report.md`: estructura y diagnostico inicial del dataset.
- `data/reports/anomaly_research_plan.md`: plan cientifico del downstream de anomalias.
- `project/src/data/`: utilidades de rutas, HDF5, particiones y datasets lazy.
- `project/src/models/`: modelos minimos de smoke para SSL y contrastive.
- `project/scripts/sh/dry_run_data.sh`: inspeccion reproducible de datos.
- `project/scripts/sh/run_ssl_smoke.sh`: smoke test SSL local en CPU.
- `project/scripts/sh/run_contrastive_smoke.sh`: smoke test contrastivo local en CPU.
- `project/scripts/sh/setup_env.sh`: crea el entorno comun Python 3.10 compatible con el servidor remoto.
- `project/scripts/sh/run_contrastive.sh`: entrenamiento contrastivo base.
- `project/scripts/sh/export_contrastive_embeddings.sh`: export de embeddings y metricas retrieval desde checkpoint.
- `project/scripts/sh/report_contrastive_run.sh`: reporte Markdown de entrenamiento/export contrastivo.
- `project/scripts/sh/diagnose_embeddings.sh`: diagnosticos PCA/kNN/prefijos sobre embeddings exportados, sin downstream.
- `tests/`: pruebas de acceso a datos y forward/loss.

## Cambios Recientes

- El modelo contrastivo ahora estandariza imagenes y espectros por muestra usando solo regiones/canales validos por mascara.
- Entrenamiento y export contrastivo registran fracciones medias de validez:
  - `image_channel_valid_fraction`;
  - `spectrum_valid_fraction`.
- `summary.json` y los reportes contrastivos nuevos separan metricas de ultima epoca y del mejor checkpoint de validacion usando campos `best_*`.
- Entrenamiento, export y reportes contrastivos nuevos incluyen diagnosticos de similitud positiva vs. negativos:
  - `positive_cosine_mean`;
  - `negative_cosine_mean`;
  - `positive_negative_margin`.
- El trainer contrastivo acepta controles configurables de estabilidad:
  - `train.gradient_clip_norm`;
  - `train.lr_scheduler=none|cosine`;
  - `train.min_learning_rate`;
  - `train.early_stopping_patience`;
  - `train.early_stopping_min_delta`.
- El modelo contrastivo ahora permite comparar encoders `simple` y `residual` mediante:
  - `model.encoder_variant`;
  - `model.encoder_width`;
  - `model.dropout`.
- El downstream de anomalias queda condicionado a tener primero un modelo contrastivo confiable, con validation estable, margen positivo-negativo sano, retrieval por encima del azar y reportes reproducibles desde `best.pt`.
- Esta mejora busca reducir sensibilidad a escala instrumental y facilitar diagnosticos de runs antes de escalar en el servidor remoto.

## Validaciones Recientes

- Local con `.venv` de `uv` Python 3.10:
  - `.venv/bin/python -m pytest -q`: `6 passed, 3 skipped`.
  - Los tests Torch se saltan localmente si Torch no esta instalado para Python 3.10.
- Servidor remoto:
  - `setup_env.sh` valida Python 3.10 + Torch `2.9.1+cu128` con CUDA disponible.
  - `.venv/bin/python -m pytest -q`: `12 passed`.
  - Smoke contrastivo con scheduler/clipping/early stopping:
    - run dir: `project/results/contrastive/scheduled_smoke_20260715`;
    - train/val: `256` / `128`;
    - epochs configuradas: `3`;
    - early stopping activo en epoca `3`;
    - `best_epoch=1`;
    - `best_val_loss=3.4658`;
    - `best_val_positive_negative_margin=0.000016`;
  - Smoke contrastivo con encoder residual:
    - run dir: `project/results/contrastive/residual_smoke_20260716`;
    - train/val: `512` / `128`;
    - `model.encoder_variant=residual`;
    - `model.encoder_width=0.5`;
    - `model.dropout=0.1`;
    - early stopping activo en epoca `3`;
    - `best_epoch=1`;
    - `best_val_loss=3.4658`;
    - `best_val_positive_negative_margin=-0.0000004`;
    - conclusion inicial: la variante residual funciona tecnicamente, pero este smoke no muestra mejora clara frente al baseline.
  - Entrenamiento contrastivo debug:
    - checkpoint: `project/results/contrastive/remote_debug_20260715_150747/best.pt`;
    - `train_loss=2.0965`;
    - `val_loss=2.0855`;
    - `val_i2s_recall@1=0.03125`;
    - `val_s2i_recall@1=0.03125`.
  - Export debug:
    - `project/results/contrastive/remote_debug_20260715_150747/export_val32/embeddings.npz`;
    - `project/results/contrastive/remote_debug_20260715_150747/export_val32/metrics.json`.
  - Diagnostico de embeddings debug:
    - `project/results/contrastive/remote_debug_20260715_150747/export_val32/diagnostics/diagnostics.json`;
    - `project/results/contrastive/remote_debug_20260715_150747/export_val32/diagnostics/pca_projection.csv`;
    - `project/results/contrastive/remote_debug_20260715_150747/export_val32/diagnostics/pca_projection.png`;
    - `pair_distance.p50=1.039301`;
    - `knn_mean_distance.p50=0.001465`.
  - Run contrastivo intermedio:
    - run dir: `project/results/contrastive/medium_20260715_161821`;
    - train/val: `8192` / `2048`;
    - epochs: `5`;
    - `train_loss=3.5533`;
    - `val_loss=4.3111`;
    - `val_i2s_recall@1=0.00098`;
    - `test_i2s_recall@1=0.00195` sobre `2048` objetos exportados;
    - observacion: la loss de train baja, pero validation empeora al final; el baseline empieza a sobreajustar o no alinear bien con esta configuracion.
  - Smoke contrastivo con normalizacion por muestra:
    - run dir: `project/results/contrastive/normalized_smoke_20260715`;
    - train/val: `512` / `128`;
    - epochs: `1`;
    - `train_loss=3.4378`;
    - `val_loss=3.4745`;
    - `val_i2s_recall@1=0.0078125`;
    - `val_s2i_recall@1=0.0078125`;
    - `val_image_channel_valid_fraction=0.52257`;
    - `val_spectrum_valid_fraction=0.21562`.
  - Run contrastivo intermedio con normalizacion por muestra:
    - run dir: `project/results/contrastive/normalized_medium_20260715_173724`;
    - train/val: `8192` / `2048`;
    - epochs: `5`;
    - mejor checkpoint por validation loss en epoca `4`;
    - summary final: `train_loss=2.9027`, `val_loss=3.1530`;
    - export validation desde mejor checkpoint: `i2s_recall@1=0.00244`, `s2i_recall@1=0.00049`;
    - export test desde mejor checkpoint: `i2s_recall@1=0.00098`, `s2i_recall@1=0.00342`;
    - comparado con el run medio anterior, la normalizacion reduce fuertemente la loss de validacion (`4.3111` a `3.1530`), pero retrieval sigue bajo y requiere mejoras de arquitectura/training.
  - Mini-sweep LR/temperatura con batch `64`:
    - sweep dir: `project/results/contrastive/sweep_20260715_175816`;
    - configs: `lr1e-4/temp0.07`, `lr3e-4/temp0.03`, `lr3e-4/temp0.07`, `lr3e-4/temp0.10`;
    - mejor validation loss dentro del sweep: `lr3e4_temp010_b64`, `best_val_loss=3.8287`;
    - mejor i2s R@1 dentro del sweep: `lr3e4_temp007_b64`, `i2s_recall@1=0.00391`;
    - conclusion: batch `64` no mejora contra el run normalizado batch `32`; el siguiente bloque deberia cambiar arquitectura/training, no solo LR/temperatura.

## Hallazgos de Datos

- Pares imagen-espectro disponibles: `64,365`.
- Espectros disponibles: `69,351`.
- Objetos solo-espectro: `4,986`.
- Pares disponibles en test filtrado: `6,586`.
- Las particiones cubren `525,508` IDs, por lo que deben filtrarse contra las keys reales de HDF5.
- Imagenes y espectros tienen mascaras importantes:
  - `img_channel_mask`;
  - `mask_spectra`.

## Estado Remoto

Servidor: privado, no versionado por seguridad.  
Directorio permitido: `~/Hackaton2026`

Dataset transferido al servidor remoto:

- `images_reduced.h5`;
- `spectra.h5`;
- particiones;
- metadata.

Estado operativo reciente: GPU libre despues de las validaciones, solo Xorg/gnome usando memoria minima. No tocar procesos fuera de `~/Hackaton2026`.

## Proximo Bloque

Siguiente bloque recomendado:

1. Mejorar el entrenamiento/modelo contrastivo base:
   - revisar capacidad de encoders;
   - probar scheduler/early stopping;
   - considerar hard negatives o memoria de negativos;
   - revisar normalizacion por canal/modalidad.
2. Enriquecer diagnosticos no-downstream con estratificacion por cobertura espectral/canales validos.
3. Exportar embeddings de validation/test filtrados desde checkpoints realmente competitivos.
4. Recien despues pasar al analisis latente orientado a anomalias.

## No Hacer Todavia

- No implementar downstream de anomalias.
- No modificar HDF5 originales.
- No lanzar jobs en el servidor remoto si hay GPU/procesos activos sin confirmacion.
- No editar codigo directamente en el servidor remoto.
