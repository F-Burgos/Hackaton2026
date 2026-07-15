# Implementation Roadmap

Fecha: 2026-07-14  
Estado: plan tecnico antes de implementar modelos.

## Principios

- Partir desde los datos finales existentes.
- No regenerar datos ni particiones.
- No cargar HDF5 completos en memoria.
- Filtrar particiones contra keys reales de los HDF5.
- Mantener test intacto para evaluacion final.
- Versionar codigo, configuraciones y reportes; no versionar datos pesados, papers ni artefactos.
- Trabajar localmente y actualizar `titae` mediante Git.

## Fase 0: Higiene y Documentacion

Estado: completada como base inicial.

Entregables:

- `Codex.md`.
- `PROJECT_STATUS.md`.
- `data/reports/dataset_structure_report.md`.
- `data/reports/anomaly_research_plan.md`.
- `.gitignore` actualizado para `hackaton/` y `Useful_articles/`.

Validacion:

- `git status`.
- `git check-ignore` para dataset y papers.
- revision manual de reportes.

## Fase 1: Estructura Minima del Proyecto

Objetivo: preparar una base reproducible sin entrenar modelos aun.

Estado: completada como estructura inicial.

Carpetas candidatas:

```text
project/
  configs/
  scripts/
    sh/
  src/
    data/
    models/
    training/
    evaluation/
    utils/
tests/
```

Entregables:

- `requirements.txt` o equivalente.
- scripts `.sh` base.
- configuracion inicial.
- guia de ejecucion inicial.

Validacion:

- import smoke test.
- comandos `.sh --help` o dry-run.

## Fase 2: Acceso Seguro a Datos

Objetivo: poder construir datasets reproducibles y pequenos smoke tests.

Estado: completada para pares imagen-espectro filtrados.

Entregables:

- utilidades para abrir HDF5 lazy.
- utilidades para listar keys:
  - imagenes;
  - espectros;
  - pares.
- utilidades para filtrar particiones.
- dataset multimodal que devuelve:
  - `object_id`;
  - `img`;
  - `img_channel_mask`;
  - `flux_lambda_normalized`;
  - `mask_spectra`;
  - `wave`.

Validacion:

- conteos iguales al reporte:
  - `64,365` pares;
  - `6,586` pares test filtrado.
- batch pequeno sin NaN/inf.
- no cargar todo en memoria.

Comandos validados:

```bash
bash project/scripts/sh/dry_run_data.sh --sample-size 1
pytest -q tests/test_data_access.py
```

## Fase 3: Baselines de Representacion

Objetivo: obtener embeddings simples antes de entrenar modelos complejos.

Estado: parcialmente iniciado con smoke models, no con baselines cientificos finales.

Candidatos:

- flatten/proyeccion PCA simple de imagen y espectro;
- estadisticas por modalidad;
- encoder aleatorio para baseline contrastivo;
- kNN/densidad sobre features simples.

Validacion:

- embeddings con IDs.
- UMAP/PCA diagnostico.
- scores kNN/densidad reproducibles.

Smoke tecnico ya validado:

```bash
bash project/scripts/sh/run_ssl_smoke.sh --modality spectrum --batch-size 4 --steps 2 --subset-size 16
bash project/scripts/sh/run_ssl_smoke.sh --modality image --batch-size 4 --steps 2 --subset-size 16
bash project/scripts/sh/run_contrastive_smoke.sh --batch-size 4 --steps 2 --subset-size 16
pytest -q tests/test_ssl_smoke.py tests/test_contrastive_smoke.py
```

## Fase 4: Modelo Contrastivo Base

Objetivo: entrenar representaciones imagen-espectro alineadas.

Estado: iniciado. Existe un trainer contrastivo base con evaluacion retrieval y checkpoint para runs pequenos/controlados.

Primera version:

- encoder imagen simple;
- encoder espectro 1D simple;
- projection heads;
- loss contrastiva global;
- metricas retrieval.

Entregables:

- entrenamiento local dry-run;
- entrenamiento completo en `titae`;
- reporte contrastivo inicial;
- checkpoint del mejor modelo.

Validacion:

- loss baja en smoke test;
- Recall@K sobre validation;
- embeddings no colapsados;
- UMAP/PCA con IDs.

Comando inicial:

```bash
bash project/scripts/sh/run_contrastive.sh
```

## Fase 5: Analisis de Espacio Latente

Objetivo: convertir embeddings en una herramienta de inspeccion cientifica.

Scores:

- kNN distance;
- densidad local;
- distancia imagen-espectro;
- cluster residual;
- neighbor consistency.

Entregables:

- ranking top-N de candidatos anomalos;
- visualizaciones;
- vecinos mas cercanos por candidato;
- comparacion por survey/campo/cobertura.

Validacion:

- revisar manualmente candidatos top-N;
- detectar si el score mide dominio en vez de anomalia.

## Fase 6: Anomalias Sinteticas

Objetivo: validar el score con anomalias controladas, sin modificar HDF5 originales.

Perturbaciones:

- imagen: patch brillante/oscuro, ruido local, corrupcion por canal valido;
- espectro: spike, bump gaussiano, segmento anomalo, escala local, ruido local.

Entregables:

- generador en memoria;
- manifest de objetos alterados;
- AUROC/AUPRC/F1/recall a FPR fijo;
- plots de distribucion de scores.

Validacion:

- no contaminar train;
- no alterar datos originales;
- no modificar canales invalidos ni puntos espectrales enmascarados.

## Fase 7: Follow-Ups

Ideas posteriores:

- crossmodal feature mapping con MLPs sobre embeddings congelados;
- memory bank de embeddings nominales;
- token-level alignment entre patches de imagen y segmentos espectrales;
- autoencoders unimodales;
- reportes HTML interactivos.

## Interfaz de Monitoreo

Primera version simple:

- `PROJECT_STATUS.md` como tablero manual.
- reportes Markdown versionados.
- logs locales y remotos en carpetas ignoradas por Git.

Version posterior:

- script `status.sh` que muestre:
  - git status;
  - ultimo commit;
  - jobs locales/remotos;
  - uso GPU en `titae`;
  - ultimos logs de entrenamiento.
