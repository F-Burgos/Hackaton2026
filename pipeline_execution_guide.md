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

Si `uv` no esta instalado en el servidor, instalarlo para el usuario antes de estos comandos.

## Proximos Comandos Pendientes

Estos scripts se agregaran cuando existan implementaciones reales:

```text
project/scripts/sh/run_ssl.sh
project/scripts/sh/run_contrastive.sh
project/scripts/sh/run_hpo.sh
project/scripts/sh/run_evaluate.sh
project/scripts/sh/run_pipeline.sh
```

Por ahora no hay entrenamiento implementado.
