# Contrastive Progress Report

Fecha: 2026-07-17

## Estado

El pipeline contrastivo ya entrena, exporta embeddings y genera diagnosticos reproducibles. Sin embargo, el modelo todavia no es suficientemente confiable para iniciar el downstream de anomalias.

La evidencia actual muestra:

- la normalizacion por muestra mejoro fuertemente la validation loss;
- los controles de entrenamiento funcionan tecnicamente;
- el margen positivo-negativo puede ser claramente mayor que cero;
- el retrieval cross-modal sigue muy bajo, cercano al azar para validation/test grandes.

## Run Controlado Mas Reciente

Run largo:

`project/results/contrastive/long_simple_20260717_134339`

Configuracion:

- encoder: `simple`;
- train/validation: `46213` / `8192`;
- batch size: `32`;
- epochs maximas: `80`;
- learning rate inicial: `0.0003`;
- scheduler: `cosine`;
- gradient clipping: `1.0`;
- early stopping patience: `8`;
- early stopping min delta: `0.002`.

Resultado entrenamiento:

| Campo | Valor |
|---|---:|
| best epoch | 3 |
| stopped early | true, epoch 11 |
| best train loss | 2.543592 |
| best val loss | 2.973322 |
| best val positive cosine mean | 0.553389 |
| best val negative cosine mean | 0.444415 |
| best val positive-negative margin | 0.108973 |
| best val i2s R@1 | 0.000732 |
| best val s2i R@1 | 0.000244 |

Export desde `best.pt`:

| Split | n | i2s R@1 | i2s R@5 | i2s R@10 | s2i R@1 | s2i R@5 | s2i R@10 | margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 8192 | 0.000732 | 0.003052 | 0.005493 | 0.000244 | 0.002441 | 0.005127 | 0.108973 |
| test | 6586 | 0.001974 | 0.005618 | 0.013969 | 0.001215 | 0.006833 | 0.012602 | 0.231702 |

## Run Controlado Anterior

Run:

`project/results/contrastive/controlled_medium_20260716_213741`

Configuracion:

- encoder: `simple`;
- train/validation: `16384` / `4096`;
- batch size: `32`;
- epochs maximas: `20`;
- learning rate inicial: `0.0003`;
- scheduler: `cosine`;
- gradient clipping: `1.0`;
- early stopping patience: `4`.

Resultado entrenamiento:

| Campo | Valor |
|---|---:|
| best epoch | 3 |
| stopped early | true, epoch 7 |
| best train loss | 2.990470 |
| best val loss | 3.116941 |
| best val positive cosine mean | 0.458725 |
| best val negative cosine mean | 0.340821 |
| best val positive-negative margin | 0.117905 |
| best val i2s R@1 | 0.000244 |
| best val s2i R@1 | 0.000488 |

Export desde `best.pt`:

| Split | n | i2s R@1 | i2s R@5 | i2s R@10 | s2i R@1 | s2i R@5 | s2i R@10 | margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 4096 | 0.000244 | 0.004395 | 0.008057 | 0.000488 | 0.003906 | 0.006836 | 0.117905 |
| test | 4096 | 0.000977 | 0.004395 | 0.009033 | 0.000733 | 0.004395 | 0.010010 | 0.118980 |

## Interpretacion

El run largo aprende una separacion positiva-negativa real: la similitud media de pares correctos supera a la similitud media de negativos por aproximadamente `0.109` en validation y `0.232` en test.

Pero el retrieval sigue debil. Esto sugiere que el modelo aprende una separacion global parcial entre pares positivos y negativos, pero no ordena de forma suficientemente precisa los vecinos cross-modal a escala de miles de objetos. Al aumentar los datos de entrenamiento, la loss mejora y el margen se sostiene, pero no aparece un salto equivalente en retrieval.

Por ahora no cumple el gate definido para downstream de anomalias.

## Decision

No implementar downstream todavia.

El siguiente bloque debe concentrarse en mejorar el modelo contrastivo o sus objetivos de entrenamiento. Prioridades candidatas:

- revisar loss contrastiva y temperatura efectiva;
- probar batches efectivos mas grandes mediante acumulacion o memoria de negativos;
- diagnosticar si los pares comparten informacion suficiente para retrieval uno-a-uno;
- analizar estratificacion por cobertura espectral/canales validos;
- comparar contra baselines simples de similitud/metadata;
- explorar encoders mas expresivos solo si mejoran validation/test de forma consistente.

## Artefactos

- `project/results/contrastive/controlled_medium_20260716_213741/summary.json`
- `project/results/contrastive/controlled_medium_20260716_213741/report_val4096.md`
- `project/results/contrastive/controlled_medium_20260716_213741/report_test4096.md`
- `project/results/contrastive/controlled_medium_20260716_213741/export_val4096/metrics.json`
- `project/results/contrastive/controlled_medium_20260716_213741/export_test4096/metrics.json`
- `project/results/contrastive/long_simple_20260717_134339/summary.json`
- `project/results/contrastive/long_simple_20260717_134339/report_val8192.md`
- `project/results/contrastive/long_simple_20260717_134339/report_test_full.md`
- `project/results/contrastive/long_simple_20260717_134339/export_val8192/metrics.json`
- `project/results/contrastive/long_simple_20260717_134339/export_test_full/metrics.json`
