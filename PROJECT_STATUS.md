# Hackaton2026 Project Status

Ultima actualizacion: 2026-07-14

## Estado Actual

Estamos en fase de definicion, inspeccion de datos y planificacion tecnica. Todavia no se han implementado modelos, dataloaders ni scripts de entrenamiento.

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

Antes de entrenar:

1. Definir estructura minima del proyecto.
2. Crear entorno reproducible.
3. Implementar utilidades de particiones filtradas.
4. Implementar datasets/dataloaders seguros para HDF5.
5. Hacer smoke tests locales.
6. Definir scripts `.sh`.
7. Preparar guia de ejecucion local/remota.

## No Hacer Todavia

- No entrenar experimentos completos.
- No implementar downstream de anomalias.
- No modificar HDF5 originales.
- No lanzar jobs en `titae` si hay GPU/procesos activos sin confirmacion.
- No editar codigo directamente en `titae`.
