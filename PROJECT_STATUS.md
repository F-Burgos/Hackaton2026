# Hackaton2026 Project Status

Ultima actualizacion: 2026-07-14

## Estado Actual

Estamos en fase de definicion, inspeccion de datos y preparacion tecnica. Ya existen utilidades de acceso lazy a datos, filtrado de particiones y smoke tests locales para SSL y contrastive. Todavia no hay entrenamiento completo, HPO ni evaluacion final.

## Decisiones Tomadas

- El dataset pesado vive en `hackaton/` y no se versiona en Git.
- Los articulos viven en `Useful_articles/` y tampoco se versionan.
- El desarrollo de codigo se hara localmente.
- Los dry runs se haran localmente cuando sea posible.
- Los experimentos completos se correran en `titae`, dentro de `~/Hackaton2026`.
- En `titae`, el codigo debe actualizarse mediante `git pull`; no se editaran archivos directamente alla.
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
- `tests/`: pruebas de acceso a datos y forward/loss.

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

Observacion: al revisar el servidor habia un experimento Ray corriendo fuera de `~/Hackaton2026`, aparentemente en `~/ATAT`, usando GPU. No tocar procesos fuera de `~/Hackaton2026`.

## Proximo Bloque

Antes de entrenar completo:

1. Convertir smoke scripts en entrenamiento formal con Hydra.
2. Definir metricas SSL y contrastive completas.
3. Definir checkpoints, MLflow y reportes HTML.
4. Agregar evaluacion contrastive/retrieval separada.
5. Preparar ejecucion remota via `git pull` en `titae`.
6. Confirmar que la GPU de `titae` este libre antes de lanzar jobs.

## No Hacer Todavia

- No entrenar experimentos completos.
- No implementar downstream de anomalias.
- No modificar HDF5 originales.
- No lanzar jobs en `titae` si hay GPU/procesos activos sin confirmacion.
- No editar codigo directamente en `titae`.
