# Contrastive HPO Report

Fecha: 2026-07-20

## Alcance

Busqueda de hiperparametros para el modelo contrastivo con batch contrastivo efectivo `128`.

Grilla:

- learning rate: `0.0001`, `0.0003`, `0.001`;
- temperatura fija InfoNCE: `0.10`, `0.15`;
- loss: `symmetric_info_nce`, `image_to_spectrum_info_nce`, `spectrum_to_image_info_nce`;
- train/validation por corrida: `16384` / `4096`;
- export validation/test por corrida: `4096` objetos.

Directorio:

`project/results/contrastive/hpo_loss_lr_temp_20260720_1320`

## Mejores Configuraciones

### Mejor balance test ranking + margen

| Run | Loss | Test margin | Test i2s median rank | Test s2i median rank | Test i2s R@1 | Test s2i R@1 |
|---|---|---:|---:|---:|---:|---:|
| `lr1e3_temp015_sym` | `symmetric_info_nce` | 0.208777 | 893 | 909 | 0.001221 | 0.001953 |
| `lr1e3_temp010_sym` | `symmetric_info_nce` | 0.156750 | 938 | 934 | 0.000732 | 0.001709 |
| `lr1e3_temp015_i2s` | `image_to_spectrum_info_nce` | 0.223026 | 881 | 998 | 0.001221 | 0.000000 |
| `lr3e4_temp010_sym` | `symmetric_info_nce` | 0.150087 | 934 | 945 | 0.001465 | 0.001465 |
| `lr3e4_temp015_sym` | `symmetric_info_nce` | 0.208957 | 933 | 949 | 0.001709 | 0.001221 |

### Mayor margen test

| Run | Loss | Test margin | Test i2s median rank | Test s2i median rank |
|---|---|---:|---:|---:|
| `lr1e3_temp015_s2i` | `spectrum_to_image_info_nce` | 0.229491 | 1086 | 930 |
| `lr1e3_temp015_i2s` | `image_to_spectrum_info_nce` | 0.223026 | 881 | 998 |
| `lr3e4_temp015_s2i` | `spectrum_to_image_info_nce` | 0.222877 | 1167 | 979 |
| `lr3e4_temp015_sym` | `symmetric_info_nce` | 0.208957 | 933 | 949 |
| `lr1e3_temp015_sym` | `symmetric_info_nce` | 0.208777 | 893 | 909 |

### Mejor validation loss

| Run | Loss | Best val loss | Test margin | Test i2s median rank | Test s2i median rank |
|---|---|---:|---:|---:|---:|
| `lr1e3_temp010_s2i` | `spectrum_to_image_info_nce` | 3.053741 | 0.136298 | 1187 | 936 |
| `lr1e3_temp015_sym` | `symmetric_info_nce` | 3.055827 | 0.208777 | 893 | 909 |
| `lr1e3_temp015_i2s` | `image_to_spectrum_info_nce` | 3.058553 | 0.223026 | 881 | 998 |
| `lr1e3_temp015_s2i` | `spectrum_to_image_info_nce` | 3.063707 | 0.229491 | 1086 | 930 |

## Interpretacion

El learning rate `0.001` fue estable en esta escala y mejora claramente varias metricas frente a `0.0003` y `0.0001`.

Las losses direccionales pueden maximizar margen o ranking en una direccion, pero introducen asimetria fuerte. Para el objetivo cientifico actual, nos interesa un espacio latente multimodal util en ambas direcciones, por lo que la configuracion mas defendible es:

```text
train.learning_rate=0.001
train.temperature=0.15
train.contrastive_loss=symmetric_info_nce
train.contrastive_accumulation_steps=4
data.batch_size=32
```

Esta configuracion no desbloquea todavia el downstream de anomalias por si sola, pero es el mejor candidato actual para un run escalado con mas datos.

## Proximo Paso

Entrenar una corrida larga con `lr1e3_temp015_sym` sobre el conjunto grande usado previamente (`46213` train, `8192` validation), exportar validation/test completo y comparar contra `long_simple_pat20_20260717_140310`.
