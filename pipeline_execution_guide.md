# Pipeline Execution Guide

Estado: guia inicial antes de implementar entrenamiento.

## Entorno Comun

El entorno comun del proyecto queda definido por lo que puede correr `titae`.
La base oficial es:

- Python `3.10`;
- `uv`;
- `.venv` dentro del repositorio;
- `--system-site-packages` para reutilizar instalaciones pesadas del sistema, especialmente PyTorch/CUDA;
- dependencias versionadas en `pyproject.toml` y `uv.lock`.

Localmente debemos imitar este entorno para dry runs. No usaremos Python 3.12/3.13 como referencia aunque puedan funcionar para algunas piezas.

Crear o actualizar el entorno:

```bash
bash project/scripts/sh/setup_env.sh
```

Si `python3` no apunta a Python 3.10, indicar el binario explicitamente:

```bash
PYTHON_BIN=/usr/bin/python3 bash project/scripts/sh/setup_env.sh
```

En `titae`, este comando debe usar `/usr/bin/python3`, que actualmente expone PyTorch/CUDA del sistema.

Activar el entorno:

```bash
source .venv/bin/activate
```

Instalar dependencias de experimentacion cuando vayamos a entrenar o registrar experimentos:

```bash
UV_CACHE_DIR=.uv-cache uv sync --group experiment
```

Instalar dependencias opcionales de visualizacion, por ejemplo UMAP:

```bash
UV_CACHE_DIR=.uv-cache uv sync --group viz
```

Verificar PyTorch/CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

En local puede imprimir `cuda False`; en `titae` debe imprimir `cuda True` antes de entrenamientos completos.
Si localmente Torch no esta disponible para Python 3.10, los tests que dependen de Torch se saltan. Los smoke tests SSL/contrastive completos deben correr en `titae`, donde Torch/CUDA esta disponible.

No usar `pip install -r requirements.txt` como flujo principal salvo emergencia. El flujo oficial es `uv` + `pyproject.toml`.

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
bash project/scripts/sh/setup_env.sh
source .venv/bin/activate
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
