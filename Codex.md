# Codex.md

## Contexto del proyecto

Este repositorio contiene el punto de partida del proyecto Hackaton2026. El trabajo debe comenzar desde los datos finales ya procesados y desde las particiones entregadas por la organización.

Codex debe respetar este alcance: si los datos finales y las particiones existen localmente, no debe volver a implementar descarga, preprocesamiento ni creación de splits, salvo que el usuario lo solicite explícitamente.

Las rutas entregadas por la organización pueden referirse a su propia máquina. En este repositorio, los archivos equivalentes viven localmente bajo `hackaton/` y deben tratarse como espejos de esos datos.

## Prioridad inmediata

Primero se debe entender y documentar cuidadosamente la estructura real de los datos disponibles:

- archivos HDF5 de imágenes y espectros;
- metadatos de generación;
- particiones;
- relación entre objetos, modalidades y splits;
- métricas descriptivas básicas;
- inconsistencias, rarezas o riesgos de leakage.

El análisis de datos debe hacerse sin cargar datasets completos en memoria. Usar inspección de headers, shapes, conteos, muestras pequeñas y lectura perezosa cuando corresponda.

## Objetivo general

El proyecto busca desarrollar y comparar:

- aprendizaje autosupervisado (`ssl`);
- aprendizaje contrastivo (`contrastive`);
- tareas downstream, aún por definir;
- baselines reproducibles.

Codex debe priorizar corrección científica, modularidad, reproducibilidad y trazabilidad.

## Alcance actual del repositorio

El repositorio Git debe versionar código, documentación, configuraciones y scripts. No debe versionar datasets completos ni artefactos pesados.

Actualmente los datos locales se encuentran bajo:

```text
hackaton/
├── dataset_metadata_reduced.json
├── images_reduced.h5
├── spectra.h5
└── partitions/
```

La estructura objetivo propuesta para el proyecto, cuando empecemos a implementarlo, es:

```text
.
├── Codex.md
├── README.md
├── pipeline_execution_guide.md
├── data/
│   └── reports/
├── project/
│   ├── configs/
│   │   ├── ssl/
│   │   ├── contrastive/
│   │   └── downstream/
│   ├── src/
│   │   ├── data/
│   │   ├── models/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── utils/
│   ├── scripts/
│   │   ├── sh/
│   │   ├── train.py
│   │   ├── hpo.py
│   │   └── evaluate.py
│   ├── reports/
│   ├── notebooks/
│   ├── results/
│   └── downstream/
├── baselines/
└── tests/
```

No crear nuevas carpetas principales sin justificación.

## Forma de trabajo

Antes de programar, Codex debe:

1. revisar la estructura, configuraciones y código existente;
2. entender la pregunta científica y el alcance del cambio;
3. indicar brevemente qué modificará y cómo lo validará;
4. preguntar antes de tomar decisiones científicas importantes.

Durante la implementación:

- hacer cambios pequeños y revisables;
- entregar reportes de progreso frecuentes, especialmente antes y después de cambios de código, ejecuciones remotas, experimentos largos o decisiones científicas;
- evitar refactors no solicitados;
- no duplicar lógica;
- separar datos, modelos, entrenamiento y evaluación;
- usar logging y errores informativos;
- no introducir dependencias sin justificación;
- mantener las rutas configurables y evitar rutas absolutas;
- no modificar datos finales silenciosamente.

Después de implementar, Codex debe:

1. ejecutar los tests relevantes;
2. ejecutar un smoke test cuando exista;
3. probar los `.sh` principales cuando existan;
4. ejecutar, cuando sea posible, un pipeline mínimo end-to-end;
5. revisar logs, métricas y artefactos;
6. corregir errores y volver a probar.

No afirmar que algo funciona si no pudo ejecutarse. Cualquier limitación por falta de GPU, datos, permisos o dependencias debe documentarse claramente.

## Datos

Reglas para trabajo con datos:

- no cargar archivos completos si son grandes;
- validar dimensiones, rangos, `NaN`, infinitos y duplicados;
- mantener identificadores estables;
- reutilizar particiones existentes cuando corresponda;
- evitar leakage entre train, validation y test;
- reservar test para evaluación final;
- usar train y validation para HPO;
- no versionar datasets completos;
- documentar cualquier inconsistencia encontrada.

La primera documentación de datos debe incluir:

- estructura de cada archivo;
- shapes, dtypes, compresión y grupos relevantes;
- conteos por modalidad;
- intersección entre imágenes y espectros;
- objetos solo con una modalidad;
- descripción de particiones;
- relación entre particiones y archivos disponibles;
- métricas descriptivas básicas de muestras pequeñas;
- rarezas o riesgos detectados.

## Artefactos requeridos para SSL

Para el mejor modelo SSL, registrar en MLflow y organizar reportes HTML en `project/reports/`:

- métricas de reconstrucción: MSE, MAE y R2 globales y por modalidad;
- métricas por región para espectros, por ejemplo líneas y continuo;
- métricas según calidad de datos: S/N, cobertura, valores válidos y fracción enmascarada;
- curvas de entrenamiento;
- distribución de errores por objeto;
- reconstrucciones representativas;
- reconstrucción vs. calidad observacional;
- espacio latente con UMAP o PCA;
- coloreos por modalidad y variables continuas disponibles;
- diagnóstico de colapso del espacio latente;
- tabla resumen;
- reporte HTML `project/reports/self_supervised_report.html`;
- checkpoint del mejor modelo.

No asumir que existen clases. Etiquetas o metadatos científicos pueden usarse para interpretación, pero no son necesarios para las métricas principales.

## Artefactos requeridos para contrastive

Para el mejor modelo contrastivo, registrar en MLflow y organizar reportes HTML en `project/reports/`:

- métricas de alineación entre pares positivos y negativos;
- retrieval cross-modal: Recall@1, Recall@5, Recall@10, median rank y MRR;
- tabla por dirección, por ejemplo `image->spectrum` y `spectrum->image`;
- comparación con baselines;
- curvas de entrenamiento;
- distribuciones de similitud;
- ejemplos cualitativos de vecinos más cercanos;
- UMAP o PCA conjunto;
- coloreos por modalidad y variables continuas disponibles;
- correspondencia entre modalidades;
- comparación antes y después del entrenamiento;
- diagnóstico de colapso;
- tabla resumen;
- reporte HTML `project/reports/contrastive_report.html`;
- checkpoint del mejor modelo;
- embeddings o vecinos solo cuando sean necesarios para los artefactos definidos.

Contrastive no debe depender de haber entrenado SSL.

## Downstream

La tarea downstream inicial será investigación de anomalías cross-modal.

Idea científica:

- entrenar primero un modelo contrastivo con pares normales de imagen y espectro;
- aprender un espacio latente compartido donde la imagen y el espectro de una misma galaxia queden próximos;
- durante evaluación final, modificar sintéticamente una fracción controlada del test set, inicialmente alrededor de 10%;
- las anomalías se inyectarán en una sola modalidad por objeto, por ejemplo agregando parches, zonas anómalas o corrupciones localizadas en imágenes o espectros;
- la otra modalidad del mismo objeto se mantiene sin modificar para actuar como referencia cross-modal;
- usar la distancia o baja similitud entre embeddings de imagen y espectro como anomaly score;
- evaluar si los objetos modificados quedan separables de los objetos no modificados.

Esta tarea no debe implementarse todavía. Primero deben quedar bien resueltos:

- inspección y documentación de datos;
- carga reproducible de pares imagen-espectro;
- particiones filtradas contra los datos disponibles;
- entrenamiento contrastivo base;
- métricas de alineación y retrieval;
- reportes y validaciones del modelo contrastivo.

Cuando se implemente downstream, se debe definir explícitamente:

- protocolo de inyección de anomalías;
- porcentaje exacto de objetos alterados;
- modalidad alterada y tipo de perturbación;
- seed y trazabilidad de objetos modificados;
- métricas de detección, por ejemplo AUROC, AUPRC y recall a FPR fijo;
- baselines de distancia o similitud;
- controles para evitar que el score detecte solo artefactos triviales de la corrupción.

Antes de implementarlas, Codex debe ayudar a definir:

- pregunta científica;
- variable objetivo;
- disponibilidad y calidad de etiquetas;
- splits válidos;
- métricas;
- baselines;
- riesgos de leakage;
- relación con embeddings SSL o contrastive.

## Hydra y configuración

Hydra debe ser la fuente de verdad para todos los valores modificables cuando exista implementación de entrenamiento.

Todos estos valores deben ser configurables:

- modo de ejecución;
- tarea;
- seed;
- rutas de datos;
- modalidad;
- batch size;
- número de workers;
- arquitectura;
- pérdidas;
- aumentaciones;
- optimizador y scheduler;
- learning rate;
- dimensiones;
- temperatura contrastiva;
- máscara de SSL;
- checkpoints;
- métricas y artefactos;
- recursos de CPU/GPU;
- fracción de GPU por entrenamiento;
- configuración de HPO.

Reglas:

- no usar rutas absolutas;
- no hardcodear hiperparámetros;
- usar composición de configuraciones;
- guardar la configuración resuelta en entrenamientos finales;
- actualizar `best.yaml` solo después de un HPO válido.

## Scripts `.sh`

Los `.sh` serán la interfaz principal. Los `.py` deben contener la lógica interna.

Estructura esperada:

```text
project/scripts/sh/
├── run_ssl.sh
├── run_contrastive.sh
├── run_hpo.sh
├── run_evaluate.sh
└── run_pipeline.sh
```

Cada `.sh` debe:

- usar `#!/usr/bin/env bash` y `set -euo pipefail`;
- detectar la raíz del proyecto;
- aceptar parámetros mediante `"$@"`;
- reenviar esos parámetros como overrides de Hydra;
- no duplicar configuraciones;
- detenerse si falla una etapa.

Debe poder ejecutarse:

- SSL o contrastive;
- imágenes, espectros o multimodal;
- entrenamiento directo sin HPO;
- HPO opcional;
- evaluación;
- pipeline completo;
- encoder aleatorio, preentrenado o desde checkpoint.

## Guía de ejecución

Codex debe crear y mantener `pipeline_execution_guide.md` en la raíz cuando existan scripts y pipeline.

La guía debe explicar:

- preparación del entorno;
- estructura de datos esperada;
- configuraciones que debe revisar el usuario;
- smoke test;
- entrenamiento SSL;
- entrenamiento contrastive;
- HPO opcional;
- entrenamiento final con `best.yaml`;
- evaluación;
- generación de reportes;
- ubicación de checkpoints y artefactos;
- uso de una GPU, varias GPUs y fracciones de GPU;
- reanudación de ejecuciones;
- errores frecuentes.

La guía debe usar principalmente comandos `.sh` que hayan sido probados.

## MLflow

Usar experimentos separados para:

- `ssl`;
- `contrastive`;
- `downstream`;
- `baselines`.

En entrenamientos finales registrar:

- configuración resuelta;
- seed;
- versión del código;
- partición;
- métricas;
- artefactos definidos;
- mejor modelo;
- tiempo y recursos.

Un entrenamiento directo sin HPO también debe registrarse.

## HPO con Optuna y Ray Tune

Valores por defecto, siempre modificables desde Hydra:

- 80 trials;
- máximo 80 épocas por trial;
- búsqueda bayesiana con Optuna;
- pruning de trials sin progreso;
- Ray Tune para paralelización y recursos.

Reglas:

- definir una única métrica objetivo;
- indicar si se maximiza o minimiza;
- no usar test;
- configurar GPU, CPU y concurrencia;
- compartir GPU solo si la memoria lo permite;
- ejecutar primero un smoke test;
- Optuna controla pruning;
- Ray controla recursos y paralelización.

Al finalizar, conservar:

- `best.yaml`;
- mejor métrica;
- mejores hiperparámetros;
- trials completos, podados y fallidos;
- espacio de búsqueda;
- seed y versión del código.

Después del HPO, reentrenar desde cero con `best.yaml` y registrar ese entrenamiento en MLflow.

Durante HPO no guardar modelos, checkpoints, embeddings completos ni gráficos por época. Solo conservar el mejor conjunto de hiperparámetros y un resumen compacto del estudio.

## SSL y contrastive

Para SSL definir:

- tarea pretexto;
- máscara o corrupción;
- encoder y decoder;
- pérdida;
- aumentaciones;
- checkpoint;
- evaluación del espacio latente.

Para contrastive definir:

- pares positivos;
- vistas;
- aumentaciones;
- negativos;
- temperatura;
- encoder;
- projection head;
- pérdida;
- evaluación de alineación y retrieval.

Validar que:

- las aumentaciones preserven la señal científica;
- no existan positivos incorrectos;
- no haya falsos negativos sistemáticos;
- los embeddings no colapsen;
- el modelo no aprenda solo variables espurias.

## Entrenamiento y evaluación

Todo entrenamiento debe:

- recibir configuración Hydra;
- validar datos y configuración;
- controlar seeds;
- soportar CPU, una GPU, varias GPUs o una fracción de GPU;
- detectar pérdidas o gradientes no finitos;
- registrar métricas;
- guardar solo checkpoints definidos;
- permitir smoke test y reanudación;
- liberar recursos al finalizar.

La evaluación debe usar un script independiente que:

1. cargue configuración y checkpoint;
2. ejecute inferencia;
3. guarde predicciones o embeddings con identificadores cuando se solicite;
4. calcule métricas;
5. genere solo los artefactos definidos;
6. cree o actualice el reporte HTML;
7. registre partición y seed.

## Notebooks, tests y Git

Los notebooks son solo para exploración y análisis. La lógica reproducible debe vivir en `project/src/`.

Agregar y ejecutar tests para:

- datasets, transformaciones, pérdidas y dimensiones;
- configuraciones;
- splits sin solapamiento;
- forward/backward;
- smoke test;
- entrenamiento sin HPO;
- evaluación desde checkpoint;
- reportes HTML;
- ejecución básica de los `.sh`;
- pipeline mínimo end-to-end.

No versionar:

- datasets;
- modelos;
- checkpoints;
- artefactos de MLflow;
- embeddings grandes;
- cachés;
- credenciales;
- archivos temporales.

Antes de un commit:

1. revisar `git status`;
2. revisar el diff;
3. detectar archivos grandes;
4. pedir confirmación explícita del usuario.

No ejecutar commits, pushes, resets, rebases ni borrados masivos sin autorización explícita.

## Uso de subagentes

Codex puede delegar tareas específicas a subagentes disponibles cuando convenga por complejidad.

El agente principal debe:

- elegir el agente según la complejidad;
- entregar contexto e instrucciones concretas;
- evitar trabajo duplicado;
- revisar propuestas y código recibido;
- integrar cambios;
- ejecutar pruebas finales;
- mantener responsabilidad sobre el resultado completo.

Ningún resultado de un subagente debe aceptarse sin validación del agente principal.

## Definición de tarea terminada

Una tarea está terminada cuando:

- respeta el alcance inicial;
- no hay hardcoding evitable;
- Hydra controla los parámetros cuando aplica;
- puede ejecutarse mediante `.sh` cuando aplica;
- funciona con o sin HPO cuando aplica;
- puede usar una GPU completa o una fracción configurable cuando aplica;
- puede comenzar desde cero o checkpoint cuando aplica;
- genera solo los artefactos solicitados;
- los reportes requeridos están creados y revisados;
- tiene tests y validaciones proporcionales al riesgo;
- el código y los scripts relevantes fueron ejecutados;
- es reproducible y no presenta leakage conocido;
- MLflow registra el entrenamiento final cuando aplica;
- `pipeline_execution_guide.md` está actualizado cuando aplica.
