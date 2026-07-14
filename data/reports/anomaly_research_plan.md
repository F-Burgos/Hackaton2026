# Multimodal Latent-Space Anomaly Research Plan

Fecha: 2026-07-14  
Estado: definicion cientifica y tecnica inicial. No implementar todavia.

## Objetivo Cientifico

Queremos estudiar si un espacio latente multimodal imagen-espectro permite identificar galaxias anomalas como objetos desalineados, aislados o inconsistentes dentro de la geometria del embedding.

La hipotesis principal es:

> Si el modelo aprende una representacion multimodal coherente, las galaxias nominales deberian formar vecindarios consistentes en el espacio latente. Las anomalías deberian aparecer como outliers globales, objetos con vecinos inconsistentes, o casos donde la imagen y el espectro no quedan bien integrados dentro del manifold multimodal.

Este foco surge de evidencia exploratoria previa: ya se han encontrado manualmente anomalias asociadas a desalineamientos entre objetos dentro de un espacio latente, aun sin haber entrenado modelos especificos para este dataset.

Por lo tanto, el objetivo principal inicial no es predecir una modalidad desde la otra, sino:

- entrenar representaciones multimodales utiles;
- estudiar la geometria del espacio latente;
- definir anomaly scores basados en vecindarios, densidad, distancia y consistencia multimodal;
- validar si las anomalias sinteticas y/o manualmente detectadas aparecen separadas.

Este downstream se implementara despues de tener resueltos:

- documentacion y validacion de datos;
- dataloaders reproducibles;
- filtrado correcto de particiones;
- entrenamiento contrastivo base;
- metricas de alineacion y retrieval;
- reportes del modelo contrastivo.

La ventaja de este enfoque es que reutiliza directamente el trabajo de las etapas anteriores. SSL, contrastive, retrieval y diagnosticos de espacio latente no son pasos descartables, sino la base del downstream: el detector de anomalias se apoya en embeddings, vecindarios, densidades y rankings ya producidos por el pipeline multimodal.

## Datos Relevantes

Del reporte `data/reports/dataset_structure_report.md`:

- pares imagen-espectro disponibles: `64,365`;
- pares disponibles en test filtrado: `6,586`;
- espectros solo-espectro: `4,986`;
- particiones originales: universo SSL de `525,508` IDs, por lo que deben filtrarse contra HDF5.

Para anomalías downstream se debe trabajar sobre el test filtrado de pares disponibles. Si se modifica aproximadamente el 10% del test, el primer protocolo tendria alrededor de `659` objetos anomalos y `5,927` controles nominales.

## Inspiracion de Literatura

### Crossmodal Feature Mapping, CVPR 2024

Archivo local:

`Useful_articles/Costanzino_Multimodal_Industrial_Anomaly_Detection_by_Crossmodal_Feature_Mapping_CVPR_2024_paper.pdf`

Ideas utiles:

- aprender relaciones cross-modal usando solo muestras nominales;
- predecir features de una modalidad desde la otra;
- usar discrepancia entre features observadas y features predichas como score de anomalia;
- combinar scores de ambas direcciones para evitar falsos positivos cuando una direccion de mapeo es ambigua;
- evitar soluciones triviales de identidad porque input y output pertenecen a modalidades distintas;
- considerar features contextualizadas, porque reducen problemas de mapeos uno-a-muchos.

Estado en este proyecto:

- queda como follow-up metodologico;
- puede ser util si el score directo en el espacio latente no captura suficientes anomalias;
- no sera el enfoque primario de la primera version.

Adaptacion posible a futuro:

- no tenemos RGB + point cloud, sino imagen astronomica multibanda + espectro 1D;
- la idea de mapeo cross-modal puede traducirse a:
  - imagen -> espectro embedding;
  - espectro -> imagen embedding;
  - score por discrepancia bidireccional;
- se podrian entrenar pequenos MLPs de mapeo entre embeddings congelados.

### Hybrid Fusion, CVPR 2023

Archivo local:

`Useful_articles/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.pdf`

Ideas utiles:

- evitar concatenar modalidades de forma ingenua si sus features tienen distribuciones distintas;
- usar aprendizaje contrastivo patch-wise o local para fomentar interaccion entre modalidades alineadas;
- mantener tambien informacion unimodal, no solo fusionada;
- combinar scores desde distintas fuentes para decision final;
- reportar metricas de deteccion y, cuando exista localizacion, metricas de segmentacion.

Adaptacion a nuestro caso:

- no tenemos alineamiento espacial pixel-a-pixel entre imagen y espectro;
- el contraste local patch-wise no se traduce directamente;
- si mas adelante hacemos token-level alignment, podria existir:
  - tokens de imagen por patch/canal;
  - tokens de espectro por segmentos de longitud de onda;
  - alineacion suave via atencion, no posicion compartida;
- para una primera version conviene evitar fusion temprana compleja y empezar con encoders separados + projection heads + loss contrastiva global.

### Adaptive Memory Broad Learning for Time Series Anomaly Detection

Archivo local:

`Useful_articles/Adaptive_Memory_Broad_Learning_System_for_Unsupervised_Time_Series_Anomaly_Detection.pdf`

Uso en este proyecto:

- no usarlo como arquitectura principal;
- usarlo para pensar anomalias en modalidad 1D tipo espectro.

Ideas utiles:

- representar senales como ventanas;
- distinguir anomalias puntuales y anomalias de segmento;
- normalizar senales antes de modelar;
- reportar AUROC, AUPRC, precision, recall y F1;
- AUPRC es especialmente importante cuando las anomalias son raras;
- considerar point adjustment solo con cuidado, porque nuestro downstream puede ser object-level, no time-point-level.

Augmentations 1D mencionadas por el paper y adaptables como perturbaciones candidatas:

- ruido gaussiano;
- modificacion forzada de puntos seleccionados;
- escalamiento de amplitud;
- inversion temporal;
- negacion.

Para espectros astronomicos, no todas son cientificamente igual de razonables. La inversion temporal y la negacion pueden servir como controles fuertes, pero no necesariamente como anomalias fisicas plausibles.

## Protocolo Downstream Propuesto

### Entrenamiento

Entrenar un modelo contrastivo o multimodal solo con datos nominales sin anomalias sinteticas:

- usar pares imagen-espectro filtrados;
- usar train/val de una particion filtrada, inicialmente fold 0;
- reservar test intacto;
- optimizar alineacion cross-modal:
  - imagen -> espectro;
  - espectro -> imagen.

El producto principal no sera solo la similitud par-a-par, sino un embedding multimodal analizable. Candidatos:

- embedding de imagen proyectado;
- embedding de espectro proyectado;
- embedding fusionado por late fusion;
- embedding conjunto producido por un modulo multimodal.

La primera version debe guardar embeddings con IDs solo cuando sea necesario para reportes y diagnosticos.

### Evaluacion Anomala

Crear una vista de evaluacion derivada del test filtrado:

- seleccionar una fraccion fija de objetos, inicialmente `10%`;
- usar seed fijo;
- modificar solo una modalidad por objeto;
- conservar la otra modalidad intacta;
- no escribir modificaciones en los HDF5 originales;
- registrar un manifest de evaluacion con:
  - object_id;
  - split;
  - modalidad alterada;
  - tipo de anomalia;
  - parametros;
  - seed.

### Anomaly Scores Iniciales

Score de alineacion par-a-par:

```text
score(object) = distance(z_image, z_spectrum)
```

Candidatos:

- distancia coseno: `1 - cosine_similarity`;
- distancia euclidiana sobre embeddings normalizados;
- score simetrico si existen projection heads direccionales.

Scores de espacio latente multimodal:

1. kNN distance:
   - distancia al k-esimo vecino en el embedding multimodal;
   - outliers deberian tener vecinos mas lejanos.

2. Local density score:
   - media o mediana de distancias a k vecinos;
   - alternativa robusta al kNN puntual.

3. Neighbor consistency score:
   - medir si vecinos cercanos comparten propiedades observacionales esperadas;
   - por ejemplo survey/prefijo/cobertura o similitud entre modalidades.

4. Modality disagreement score:
   - distancia entre embedding de imagen y embedding de espectro del mismo objeto;
   - util como componente, no necesariamente como score unico.

5. Cluster residual:
   - distancia al centroide de cluster o componente local;
   - util despues de explorar UMAP/PCA/k-means/HDBSCAN.

Scores de follow-up:

- discrepancia imagen -> espectro via MLP de mapeo;
- discrepancia espectro -> imagen via MLP de mapeo;
- producto o media geometrica de discrepancias bidireccionales;
- calibracion por prefijo/survey para evitar que el score mida dominio en lugar de anomalia.

## Anomalias Sinteticas Candidatas

### Imagen

Perturbaciones iniciales:

1. Patch brillante localizado:
   - agregar un parche cuadrado o gaussiano en uno o varios canales validos;
   - amplitud relativa a estadisticas del objeto o canal.

2. Patch oscuro o dropout:
   - atenuar una region localizada;
   - simula perdida local o artefacto de sustraccion.

3. Ruido localizado:
   - agregar ruido gaussiano o impulsivo en una region;
   - mantener region pequena para no hacer trivial la deteccion.

4. Corrupcion por canal:
   - alterar solo un canal valido;
   - importante porque las imagenes tienen `img_channel_mask` variable.

Controles:

- no modificar canales invalidos;
- no cambiar `img_channel_mask`;
- limitar magnitud para que el problema no sea visualmente trivial;
- registrar mascara de anomalia si se quiere evaluar localizacion mas adelante.

### Espectro

Perturbaciones iniciales inspiradas por anomalias 1D/time-series:

1. Spike puntual:
   - aumentar o disminuir uno o pocos puntos validos;
   - equivalente a anomalia puntual.

2. Segmento anomalo:
   - modificar un intervalo contiguo de longitud de onda;
   - equivalente a anomalia de segmento.

3. Bump gaussiano:
   - agregar una emision/absorcion sintetica localizada;
   - mas astronomicamente plausible que modificar puntos aislados.

4. Escalamiento local:
   - multiplicar un segmento por un factor;
   - inspirado por augmentation de escala.

5. Ruido local:
   - aumentar ruido en un intervalo;
   - util para simular baja calidad o contaminacion.

Controles:

- modificar solo posiciones con `mask_spectra == 1`;
- no cambiar `wave`;
- no cambiar `mask_spectra`, salvo que se defina una anomalia de perdida de cobertura;
- evitar invertir o negar espectros en el protocolo principal porque pueden ser demasiado artificiales;
- usar inversion/negacion solo como stress tests o controles.

## Metricas

Object-level:

- AUROC;
- AUPRC;
- F1 en threshold seleccionado sobre validation sintetico o por criterio predefinido;
- precision/recall;
- recall a FPR fijo, por ejemplo 1%, 5% y 10%;
- distribucion de scores nominales vs anomalos.

Exploracion del espacio latente:

- UMAP/PCA/t-SNE para inspeccion cualitativa;
- histogramas de kNN distance;
- distribuciones de densidad local;
- visualizacion de vecinos mas cercanos para objetos con score alto;
- ranking top-N de candidatos anomalos;
- comparacion entre anomalias sinteticas y anomalias manuales conocidas, si se registran.

Por subconjunto:

- anomalia en imagen vs anomalia en espectro;
- HST vs JWST;
- prefijo/campo;
- cantidad de canales validos;
- cobertura espectral;
- tipo y magnitud de perturbacion.

Contrastive previo:

- Recall@K image->spectrum;
- Recall@K spectrum->image;
- median rank;
- MRR;
- similitud positiva vs negativa.

## Baselines

Baselines minimos:

1. Random score.
2. Distancia entre embeddings de encoders no entrenados.
3. Distancia contrastiva despues de entrenamiento.
4. kNN/densidad sobre embeddings unimodales.
5. Score unimodal simple:
   - norma o cambio de estadisticas de imagen;
   - estadisticas de espectro como outlier baseline.

Baselines posteriores:

- nearest-neighbor/memory bank en embeddings nominales;
- MLP cross-modal mapping congelando encoders contrastivos;
- autoencoder unimodal por modalidad.

## Riesgos Cientificos

1. Perturbaciones demasiado faciles.
   - El modelo podria detectar artefactos obvios, no incoherencia cross-modal.

2. Perturbaciones no plausibles.
   - Especialmente en espectros, negacion o inversion pueden no representar anomalias astronomicas utiles.

3. Confusion por dominio.
   - HST/JWST y distintos campos tienen distribuciones diferentes. El score podria medir dominio.

4. Cobertura y mascaras.
   - Espectros con muy baja cobertura pueden generar embeddings menos confiables.

5. Canal faltante en imagen.
   - Alterar canales invalidos no tendria sentido y podria introducir bugs silenciosos.

6. Test contamination.
   - Las anomalias sinteticas deben existir solo como vista de evaluacion, nunca sobrescribir datos ni contaminar training.

7. Interpretacion del embedding.
   - Una separacion en UMAP no implica necesariamente anomalia fisica; se debe contrastar con vecinos, metadata y ejemplos visuales.

## Primera Version Recomendada

Cuando llegue el momento de implementar:

1. Crear dataloaders filtrados por particion y disponibilidad real.
2. Entrenar modelo contrastivo global imagen-espectro.
3. Verificar retrieval y alineacion en validation.
4. Extraer embeddings de train/val/test filtrados.
5. Analizar espacio latente:
   - UMAP/PCA;
   - kNN distance;
   - densidad local;
   - ejemplos top-N.
6. Crear generador de anomalias en memoria para test:
   - 5% imagen patch;
   - 5% espectro bump/segmento;
   - total 10% anomalos.
7. Comparar scores:
   - distancia imagen-espectro;
   - kNN/densidad en embedding multimodal;
   - combinacion simple de ambos.
8. Reportar AUROC, AUPRC, recall a FPR fijo, rankings y distribuciones.

Esta version evita fusion compleja y permite validar rapidamente la hipotesis cientifica: si las anomalias se manifiestan primero como geometria rara en el espacio latente, antes de invertir en mapeos cross-modal mas sofisticados.

## No Implementar Todavia

No implementar este downstream hasta que esten listos:

- reporte de datos validado;
- estructura de proyecto;
- entorno reproducible;
- dataloaders;
- scripts `.sh`;
- entrenamiento contrastivo base;
- metricas contrastivas;
- plan de ejecucion local/remoto.
