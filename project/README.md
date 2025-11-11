# NMF + kNN Digits Classifier with FastAPI

Proyecto en Python que entrena NMF desde cero (ALS + Gradiente Proyectado con Armijo) sobre `load_digits()` y expone una API con FastAPI. Incluye una UI en `/ui` y, opcionalmente, un clasificador general de objetos (`/predict_imagenet`).

## Estructura

- `app/`
  - `api.py` — API FastAPI (`/predict`, `/predict_imagenet`, `/health`, `/ui`).
  - `nmf_core.py` — Implementación de `pg_nnls` y `als_pg` (NMF desde cero).
  - `ui.html` — Interfaz HTML (drag & drop) que consume la API y muestra resultados.
- `scripts/`
  - `train_nmf.py` — Entrena NMF (r=32), kNN (k=3), evalúa, y guarda artefactos.
- `requirements.txt` — Dependencias.
- Artefactos (se generan tras entrenar, en la raíz de `project/`):
  - `nmf_artifacts.npz` — Matriz `W` e `img_shape`.
  - `knn.joblib` — Clasificador kNN entrenado sobre `H`.

## Instalación

1) Crear y activar un entorno virtual (recomendado)
2) Instalar dependencias:

```
pip install -r requirements.txt
```

Para usar el clasificador de objetos (`/predict_imagenet`), instala además PyTorch/torchvision (opcional; CPU):

```
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Entrenamiento

Desde la carpeta `project/`:

```
python -m scripts.train_nmf
```

Esto entrena NMF (ALS-PG, r=32, 25 iteraciones), entrena kNN (k=3), evalúa accuracy y guarda `nmf_artifacts.npz` y `knn.joblib` en la raíz de `project/`.

## Ejecutar API

Desde `project/`:

```
python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8123
```

- Salud: http://127.0.0.1:8123/health
- Docs (Swagger): http://127.0.0.1:8123/docs
- UI: http://127.0.0.1:8123/ui

Si ejecutas desde la raíz del repo, usa: `python -m uvicorn project.app.api:app --reload --host 127.0.0.1 --port 8123`.

## Uso rápido

Predicción de dígitos (8×8, escala de grises):

```
curl -X POST -F "image=@path/to/image.png" http://127.0.0.1:8123/predict
```

La API:
- Convierte a escala de grises, redimensiona a 8×8, normaliza [0,1]
- Proyecta en `W` vía PG-NNLS
- Predice con kNN (k=3)
- Devuelve `{ "pred": <dígito>, "seen_8x8": [[...],[...],...] }`

Clasificador general (opcional, requiere torch/torchvision instalados):

```
curl -X POST -F "image=@path/to/photo.jpg" http://127.0.0.1:8123/predict_imagenet
```

Devuelve top-1 y top-5 etiquetas de ImageNet con probabilidades.

## Notas

- La API busca los artefactos respecto a la raíz del proyecto, por lo que funcionan sin importar el directorio desde el que lances Uvicorn.
- Si no instalas PyTorch/torchvision, `/predict_imagenet` devolverá un error 500 informativo; `/predict` funciona sin esos paquetes.
- El dataset `digits` se normaliza dividiendo por 16; las imágenes subidas se normalizan dividiendo por 255.
