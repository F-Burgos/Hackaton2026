# Hackaton2026 Project Status

Ultima actualizacion: 2026-07-15

## Estado Actual

Estamos en fase de preparacion tecnica y entrenamiento contrastivo base. Ya existen utilidades de acceso lazy a datos, filtrado de particiones, smoke tests SSL/contrastive, entrenamiento contrastivo base, metricas retrieval, export de embeddings por split y diagnosticos ligeros del espacio latente. Todavia no hay HPO, entrenamiento largo, reportes cientificos finales ni downstream de anomalias implementado.

## Decisiones Tomadas

- El dataset pesado vive en `hackaton/` y no se versiona en Git.
- Los articulos viven en `Useful_articles/` y tampoco se versionan.
- Outputs livianos de evidencia bajo `project/results/` (`.md`, `.json`, `.jsonl`, `.csv`, `.png`) se versionan; checkpoints, embeddings `.npz`, logs crudos y `outputs/` runtime se ignoran.
- `CLAUDE.md` queda como referencia local ignorada por Git; el archivo operativo versionado es `Codex.md`.
- El desarrollo de codigo se hara localmente.
- Los dry runs se haran localmente cuando sea posible.
- Los experimentos completos se correran en `titae`, dentro de `~/Hackaton2026`.
- En `titae`, el codigo debe actualizarse mediante `git pull`; no se editaran archivos directamente alla.
- El entorno comun del proyecto sigue lo que `titae` puede ejecutar: Python 3.10, `uv`, `.venv` local al repo y `--system-site-packages` para reutilizar PyTorch/CUDA del sistema.
- El downstream principal sera investigacion de anomalias en el espacio latente multimodal.
- Crossmodal feature mapping queda como follow-up, no como primera version.

## Artefactos Creados

- `Codex.md`: instrucciones de workflow del proyecto.
- `data/reports/dataset_structure_report.md`: estructura y diagnostico inicial del dataset.
- `data/reports/anomaly_research_plan.md`: plan cientifico del downstream de anomalias.
- `project/src/data/`: utilidades de rutas, HDF5, particiones y datasets lazy.
- `project/src/models/`: modelos minimos de smoke para SSL y contrastive.
- `project/scripts/sh/dry_run_data.sh`: inspeccion reproducible de datos.
- `project/scripts/sh/run_ssl_smoke.sh`: smoke test SSL local en CPU.
- `project/scripts/sh/run_contrastive_smoke.sh`: smoke test contrastivo local en CPU.
- `project/scripts/sh/setup_env.sh`: crea el entorno comun Python 3.10 compatible con `titae`.
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
- Esta mejora busca reducir sensibilidad a escala instrumental y facilitar diagnosticos de runs antes de escalar en `titae`.

## Validaciones Recientes

- Local con `.venv` de `uv` Python 3.10:
  - `.venv/bin/python -m pytest -q`: `5 passed, 3 skipped`.
  - Los tests Torch se saltan localmente si Torch no esta instalado para Python 3.10.
- `titae`:
  - `setup_env.sh` valida Python 3.10 + Torch `2.9.1+cu128` con CUDA disponible.
  - `pytest -q`: `8 passed`.
  - Entrenamiento contrastivo debug:
    - checkpoint: `project/results/contrastive/titae_debug_20260715_150747/best.pt`;
    - `train_loss=2.0965`;
    - `val_loss=2.0855`;
    - `val_i2s_recall@1=0.03125`;
    - `val_s2i_recall@1=0.03125`.
  - Export debug:
    - `project/results/contrastive/titae_debug_20260715_150747/export_val32/embeddings.npz`;
    - `project/results/contrastive/titae_debug_20260715_150747/export_val32/metrics.json`.
  - Diagnostico de embeddings debug:
    - `project/results/contrastive/titae_debug_20260715_150747/export_val32/diagnostics/diagnostics.json`;
    - `project/results/contrastive/titae_debug_20260715_150747/export_val32/diagnostics/pca_projection.csv`;
    - `project/results/contrastive/titae_debug_20260715_150747/export_val32/diagnostics/pca_projection.png`;
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

Servidor: `felipeiburgos@titae.inf.udec.cl`  
Directorio permitido: `~/Hackaton2026`

Dataset transferido a `titae`:

- `images_reduced.h5`;
- `spectra.h5`;
- particiones;
- metadata.

Estado operativo reciente: GPU libre despues de las validaciones, solo Xorg/gnome usando memoria minima. No tocar procesos fuera de `~/Hackaton2026`.

## Proximo Bloque

Siguiente bloque recomendado:

1. Correr un nuevo smoke/run contrastivo pequeno con normalizacion por muestra:
   - learning rate;
   - temperatura;
   - regularizacion;
   - batch size / negativos.
2. Enriquecer diagnosticos no-downstream con estratificacion por cobertura espectral/canales validos.
3. Exportar embeddings de validation/test filtrados desde checkpoints realmente competitivos.
4. Recien despues pasar al analisis latente orientado a anomalias.

## No Hacer Todavia

- No implementar downstream de anomalias.
- No modificar HDF5 originales.
- No lanzar jobs en `titae` si hay GPU/procesos activos sin confirmacion.
- No editar codigo directamente en `titae`.
