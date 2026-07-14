# Pipeline Execution Guide

Estado: guia inicial antes de implementar entrenamiento.

## Entorno Local

Se recomienda usar `uv`.

Crear entorno virtual local:

```bash
UV_CACHE_DIR=.uv-cache uv venv --system-site-packages
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv sync
```

`--system-site-packages` permite reutilizar instalaciones pesadas ya presentes, especialmente PyTorch/CUDA, sin reinstalarlas desde cero.
El entorno base no instala PyTorch por defecto; antes de entrenar hay que verificar si existe una build CUDA compatible en la maquina o instalarla explicitamente dentro de `.venv`.

Instalar dependencias de experimentacion cuando vayamos a entrenar o registrar experimentos:

```bash
UV_CACHE_DIR=.uv-cache uv sync --group experiment
```

Instalar dependencias opcionales de visualizacion, por ejemplo UMAP:

```bash
UV_CACHE_DIR=.uv-cache uv sync --group viz
```

Instalar dependencias de desarrollo:

```bash
UV_CACHE_DIR=.uv-cache uv sync --group dev
```

Alternativa con `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si el entorno ya existe:

```bash
source .venv/bin/activate
```

## Verificar Estado Local

```bash
bash project/scripts/sh/status.sh
```

Este comando muestra:

- estado Git;
- archivos de datos visibles;
- dependencias Python basicas.

## Dry Run de Datos

Validar HDF5, particiones filtradas y una muestra pequena:

```bash
bash project/scripts/sh/dry_run_data.sh --sample-size 2
```

Salida esperada principal:

- `images: 64365`
- `spectra: 69351`
- `paired: 64365`
- `test: 6586`
- `dry-run OK`

## Smoke Tests Locales

Estos comandos son pruebas tecnicas pequenas en CPU. No son entrenamientos cientificos finales.

SSL espectral:

```bash
bash project/scripts/sh/run_ssl_smoke.sh \
  --modality spectrum \
  --batch-size 4 \
  --steps 2 \
  --subset-size 16
```

SSL imagen:

```bash
bash project/scripts/sh/run_ssl_smoke.sh \
  --modality image \
  --batch-size 4 \
  --steps 2 \
  --subset-size 16
```

Contrastive imagen-espectro:

```bash
bash project/scripts/sh/run_contrastive_smoke.sh \
  --batch-size 4 \
  --steps 2 \
  --subset-size 16
```

Pruebas:

```bash
pytest -q
```

## Datos Esperados

Los datos locales deben existir bajo:

```text
hackaton/
  images_reduced.h5
  spectra.h5
  dataset_metadata_reduced.json
  partitions/v1/ssl/
```

Los datos no se versionan en Git.

## Servidor `titae`

Reglas:

- usar solo `~/Hackaton2026`;
- actualizar codigo con `git pull`;
- no editar directamente en el servidor;
- no lanzar jobs si ya hay experimentos activos sin confirmacion.

Verificar estado remoto:

```bash
bash project/scripts/sh/check_remote.sh
```

El entorno remoto debe crearse dentro de `~/Hackaton2026`:

```bash
cd ~/Hackaton2026
UV_CACHE_DIR=.uv-cache uv venv --system-site-packages
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv sync --group experiment
```

En `titae`, `uv` esta disponible en:

```text
/home/felipeiburgos/.local/bin/uv
```

Si no aparece en una sesion no interactiva, agregar temporalmente:

```bash
export PATH=$HOME/.local/bin:$PATH
```

Antes de entrenar, confirmar PyTorch/CUDA:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Proximos Comandos Pendientes

Estos scripts se agregaran cuando existan implementaciones de entrenamiento/evaluacion completas:

```text
project/scripts/sh/run_ssl.sh
project/scripts/sh/run_contrastive.sh
project/scripts/sh/run_hpo.sh
project/scripts/sh/run_evaluate.sh
project/scripts/sh/run_pipeline.sh
```

Por ahora no hay entrenamiento implementado.
