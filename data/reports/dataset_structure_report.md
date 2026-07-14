# Dataset Structure Report

Fecha de inspeccion: 2026-07-14  
Repositorio local: `/home/pipe/Hackaton2026`  
Datos inspeccionados: `hackaton/`

## Resumen Ejecutivo

El dataset local contiene dos archivos HDF5 principales:

- `hackaton/images_reduced.h5`: imagenes reducidas, solo para objetos que tambien tienen espectro.
- `hackaton/spectra.h5`: espectros completos del universo espectroscopico disponible.

El conjunto util para entrenamiento multimodal directo contiene `64,365` pares imagen-espectro. El archivo de espectros contiene `69,351` objetos, por lo que hay `4,986` objetos con espectro pero sin imagen. No hay objetos con imagen sin espectro.

La rareza principal esta en las particiones: los JSON de particiones describen un universo SSL de `525,508` IDs, pero los HDF5 reducidos contienen solo `64,365` pares disponibles. Por lo tanto, cualquier dataset/dataloader debe filtrar las particiones contra las keys realmente presentes en los HDF5 antes de entrenar o evaluar.

## Archivos

| Archivo | Tamano | Rol |
|---|---:|---|
| `hackaton/images_reduced.h5` | 3,325,588,154 bytes | Stamps multibanda de galaxias |
| `hackaton/spectra.h5` | 764,531,449 bytes | Espectros normalizados |
| `hackaton/dataset_metadata_reduced.json` | 2,081 bytes | Metadata del dataset reducido |
| `hackaton/partitions/v1/manifest.json` | 566 bytes | Configuracion de particiones |
| `hackaton/partitions/v1/ssl/metadata.json` | 598 bytes | Metadata de splits SSL |
| `hackaton/partitions/v1/ssl/fold_0.json` ... `fold_4.json` | 22,699,411 bytes cada uno | Folds train/validation |
| `hackaton/partitions/v1/ssl/test.json` | 2,520,424 bytes | Test holdout |

Los datos pesados estan ignorados por Git mediante `.gitignore`.

## Metadata Declarada

`dataset_metadata_reduced.json` declara:

- `run_id`: `531001e55e`
- `version`: `v1`
- `n_img`: `64,365`
- `n_spec`: `69,351`
- `n_pairs`: `64,365`
- `spec_len`: `739`
- shapes de imagen:
  - `[5, 64, 64]`: `52,723`
  - `[9, 64, 64]`: `11,642`

Tambien declara clases `non_clumpy -> 0` y `clumpy -> 1`, pero los conteos efectivos de clasificacion son cero:

- `cls_img`: `0`
- `cls_spec`: `0`

Para el alcance actual, no se debe asumir que existen etiquetas downstream listas.

## Estructura de `images_reduced.h5`

Cada objeto es un grupo HDF5 con nombre de ID estable. Todos los grupos tienen exactamente:

```text
<object_id>/
  img
  img_channel_mask
```

Estructura exacta observada:

| Dataset | Shape | Dtype | Compresion | Conteo |
|---|---:|---|---|---:|
| `img` | `(5, 64, 64)` | `float32` | `lzf` | `52,723` |
| `img` | `(9, 64, 64)` | `float32` | `lzf` | `11,642` |
| `img_channel_mask` | `(5,)` | `float32` | ninguna | `52,723` |
| `img_channel_mask` | `(9,)` | `float32` | ninguna | `11,642` |

No se encontraron grupos de imagen con mascara de canales completamente cero.

### Canales Validos en Imagenes

Distribucion exacta de cantidad de canales validos por objeto:

| Canales validos | Objetos |
|---:|---:|
| 1 | 270 |
| 2 | 1,043 |
| 3 | 4,538 |
| 4 | 10,262 |
| 5 | 41,768 |
| 6 | 245 |
| 7 | 2,836 |
| 8 | 872 |
| 9 | 2,531 |

Por shape:

| Shape | Canales validos | Objetos |
|---|---:|---:|
| `(5, 64, 64)` | 2 | 45 |
| `(5, 64, 64)` | 3 | 3,341 |
| `(5, 64, 64)` | 4 | 9,917 |
| `(5, 64, 64)` | 5 | 39,420 |
| `(9, 64, 64)` | 1 | 270 |
| `(9, 64, 64)` | 2 | 998 |
| `(9, 64, 64)` | 3 | 1,197 |
| `(9, 64, 64)` | 4 | 345 |
| `(9, 64, 64)` | 5 | 2,348 |
| `(9, 64, 64)` | 6 | 245 |
| `(9, 64, 64)` | 7 | 2,836 |
| `(9, 64, 64)` | 8 | 872 |
| `(9, 64, 64)` | 9 | 2,531 |

Implicacion: los modelos de imagen deben usar `img_channel_mask`. No basta con asumir que todos los canales estan presentes.

## Estructura de `spectra.h5`

Cada objeto es un grupo HDF5 con nombre de ID estable. Todos los grupos tienen exactamente:

```text
<object_id>/
  flux_lambda_normalized
  mask_spectra
  wave
```

Estructura exacta observada:

| Dataset | Shape | Dtype | Compresion | Conteo |
|---|---:|---|---|---:|
| `flux_lambda_normalized` | `(739,)` | `float32` | `lzf` | `69,351` |
| `mask_spectra` | `(739,)` | `float32` | ninguna | `69,351` |
| `wave` | `(739,)` | `float32` | ninguna | `69,351` |

Todas las grillas de longitud de onda inspeccionadas tienen:

- primer valor: `1500.0`
- ultimo valor: `9482.3935546875`

La inspeccion exacta de extremos encontro un unico patron `(1500.0, 9482.3935546875)` en los `69,351` espectros.

### Cobertura Espectral

La mascara `mask_spectra` indica puntos validos. No se encontraron espectros con mascara completamente cero.

Resumen exacto de cantidad de puntos validos:

| Metrica | Puntos validos |
|---|---:|
| min | 1 |
| p01 | 13 |
| p05 | 40 |
| p25 | 131 |
| mediana | 202 |
| media | 247.90 |
| p75 | 214 |
| p95 | 739 |
| p99 | 739 |
| max | 739 |

Bins de cobertura:

| Puntos validos | Espectros |
|---|---:|
| 0 | 0 |
| 1-49 | 4,862 |
| 50-99 | 7,672 |
| 100-199 | 20,857 |
| 200-399 | 23,891 |
| 400-738 | 7,820 |
| 739 | 4,249 |

Implicacion: la cobertura espectral es muy variable. El pipeline debe usar `mask_spectra` en losses, metrics y modelos. Para reportes SSL conviene estratificar por cobertura o fraccion enmascarada.

## Pares Imagen-Espectro

Conteos exactos por keys HDF5:

| Conjunto | Conteo |
|---|---:|
| imagenes | 64,365 |
| espectros | 69,351 |
| interseccion imagen-espectro | 64,365 |
| solo espectro | 4,986 |
| solo imagen | 0 |

Implicacion: para contrastive multimodal se debe usar la interseccion de keys. Para SSL unimodal de espectros, podria usarse el conjunto completo de `69,351` espectros si esa decision se documenta.

## Conteos por Survey/Campo

Conteos en `images_reduced.h5`:

| Prefijo | Imagenes |
|---|---:|
| `hst_3dhst_uds` | 12,722 |
| `hst_3dhst_goodss` | 11,368 |
| `hst_3dhst_aegis` | 9,862 |
| `hst_3dhst_cosmos` | 9,426 |
| `hst_3dhst_goodsn` | 9,345 |
| `jwst_jades` | 3,888 |
| `jwst_uds` | 2,987 |
| `jwst_ceers` | 2,732 |
| `jwst_cosmos` | 2,035 |

Conteos en `spectra.h5`:

| Prefijo | Espectros |
|---|---:|
| `hst_3dhst_uds` | 13,647 |
| `hst_3dhst_goodss` | 11,813 |
| `hst_3dhst_aegis` | 10,479 |
| `hst_3dhst_cosmos` | 10,152 |
| `hst_3dhst_goodsn` | 9,950 |
| `jwst_jades` | 4,666 |
| `jwst_ceers` | 3,511 |
| `jwst_uds` | 3,068 |
| `jwst_cosmos` | 2,065 |

La mayoria de los pares disponibles son HST. El conjunto JWST existe pero es menor, por lo que conviene monitorear metricas por prefijo/survey.

## Particiones

`hackaton/partitions/v1/manifest.json` declara:

- `random_seed`: `42`
- `n_folds`: `5`
- `z_min_train`: `0.7`
- bins de redshift: `[0.7, 1.0, 1.5, 2.0, 2.5, 3.5, 5.5, 9.0]`
- `cross_match_tol_arcsec`: `0.5`
- `valid_oids_path`: `data/base_quality/37542700c9/valid_oids.json`

`hackaton/partitions/v1/ssl/metadata.json` declara:

- universo: `ssl`
- total: `525,508`
- test: `52,543`
- train/validation: `472,965`
- estratificacion: `z_bin_and_groups`

Los folds son cross-validation sobre el conjunto train/validation. En cada fold, train y validation no se solapan. Test tampoco se solapa con train/validation.

### Disponibilidad Real Dentro de HDF5 Reducidos

Las particiones contienen IDs de un universo mucho mayor que el HDF5 reducido. Estos son los conteos disponibles despues de intersectar con las keys de pares imagen-espectro:

| Split | Total IDs particion | Pares disponibles |
|---|---:|---:|
| fold 0 train | 378,303 | 46,213 |
| fold 0 val | 94,662 | 11,566 |
| fold 1 train | 378,387 | 46,332 |
| fold 1 val | 94,578 | 11,447 |
| fold 2 train | 378,503 | 46,237 |
| fold 2 val | 94,462 | 11,542 |
| fold 3 train | 378,491 | 46,179 |
| fold 3 val | 94,474 | 11,600 |
| fold 4 train | 378,176 | 46,155 |
| fold 4 val | 94,789 | 11,624 |
| test | 52,543 | 6,586 |

Para `test.json`, el desglose disponible es:

| Grupo | IDs test | En imagenes | En espectros | Pares disponibles |
|---|---:|---:|---:|---:|
| hst | 14,351 | 5,423 | 5,756 | 5,423 |
| jwst | 38,192 | 1,163 | 1,320 | 1,163 |
| total | 52,543 | 6,586 | 7,076 | 6,586 |

No se detectaron duplicados dentro de los splits inspeccionados.

## Estadisticas Numericas de Muestra

Para evitar cargar los HDF5 completos en memoria, se inspecciono una muestra deterministica de `250` pares distribuidos en las keys ordenadas.

En esa muestra:

- imagenes: `0` NaN y `0` infinitos;
- espectros, mascaras y longitudes de onda: `0` NaN y `0` infinitos;
- canales de imagen ausentes en la muestra: `201` canales en total;
- fraccion valida espectral media: `0.319`;
- fraccion valida espectral mediana: `0.274`.

Resumen de valores de imagen por objeto en la muestra:

| Estadistica por objeto | min global | mediana | media | max global |
|---|---:|---:|---:|---:|
| minimo de imagen | -10.247 | -9.017 | -7.483 | -0.223 |
| maximo de imagen | 0.556 | 9.782 | 8.738 | 13.504 |
| media de imagen | -0.070 | 0.323 | 0.458 | 3.571 |
| std de imagen | 0.037 | 4.239 | 3.464 | 4.765 |

Resumen de valores de flujo valido por objeto en la muestra:

| Estadistica por objeto | min global | mediana | media | max global |
|---|---:|---:|---:|---:|
| minimo de flujo valido | -7.377 | -0.855 | -0.805 | 0.880 |
| maximo de flujo valido | -0.440 | 1.481 | 1.672 | 6.385 |
| media de flujo valido | -0.998 | 0.728 | 0.626 | 1.182 |
| std de flujo valido | 0.001 | 0.506 | 0.542 | 1.721 |

Estas estadisticas son diagnosticas, no sustituyen normalizacion o validaciones del pipeline.

## Rarezas y Riesgos Detectados

1. Las particiones son para un universo mucho mayor que los archivos HDF5 reducidos.  
   Accion requerida: filtrar cada split contra las keys disponibles antes de construir datasets.

2. Hay `4,986` objetos solo-espectro.  
   Accion requerida: decidir explicitamente si se usan para SSL espectral unimodal o si se excluyen para mantener comparabilidad multimodal.

3. Las imagenes tienen cantidad variable de canales validos.  
   Accion requerida: usar `img_channel_mask` siempre; no asumir que los canales faltantes son datos reales.

4. Los espectros tienen cobertura muy variable.  
   Accion requerida: usar `mask_spectra` en entrenamiento/evaluacion y reportar metricas por cobertura.

5. El dataset reducido no contiene labels downstream efectivos, aunque la metadata menciona `clumpy`/`non_clumpy`.  
   Accion requerida: no disenar downstream supervisado sin confirmar etiquetas adicionales.

6. Los prefijos HST y JWST estan desbalanceados.  
   Accion requerida: reportar metricas por survey/campo, especialmente en contrastive y retrieval.

## Recomendaciones para el Pipeline

- Definir una utilidad central para cargar keys disponibles:
  - `image_keys`
  - `spectrum_keys`
  - `paired_keys = image_keys & spectrum_keys`
- Definir una utilidad central para filtrar particiones por disponibilidad real.
- Para contrastive multimodal, usar solo `paired_keys`.
- Para test de anomalías futuro, partir desde los `6,586` pares disponibles del split test filtrado.
- Mantener test intacto hasta la evaluacion final. Las anomalías sintéticas deben inyectarse solo como vista/evaluacion controlada, no modificando los HDF5 originales.
- Registrar siempre el ID original del objeto y la modalidad usada.
- En reportes, estratificar al menos por:
  - `hst` vs `jwst`;
  - prefijo/campo;
  - cantidad de canales validos;
  - fraccion valida del espectro.

## Comandos de Inspeccion Usados

La inspeccion se hizo con Python, `h5py`, `json` y lectura perezosa de HDF5. No se cargaron datasets completos en memoria. Los pases exactos leyeron:

- keys HDF5;
- shapes y dtypes;
- `img_channel_mask`;
- `mask_spectra`;
- extremos de `wave`;
- archivos JSON de particiones.

La muestra numerica leyo `250` pares completos para estadisticas diagnosticas.
