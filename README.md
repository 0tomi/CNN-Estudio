# CNN · Cuaderno experimental

Laboratorio local para estudiar CNN con **entrenamiento real en CPU**, arquitectura editable, progreso por batch, test held-out y mapas de interpretabilidad. La ruta rápida usa Digits: 1797 imágenes de 8×8 sin descargas.

## Abrir el laboratorio

```bash
uv sync --python 3.12
uv run streamlit run app.py --server.port 8501
```

Abrí **http://localhost:8501**. Elegí CNN base y presioná **Entrenar CNN**. La primera ejecución estándar dura pocos segundos en CPU; otras configuraciones requieren más tiempo. MNIST/FashionMNIST descargan datos al prepararlos/entrenarlos y respetan el test oficial. Digits no requiere internet.

Si no tenés uv: instalalo desde su documentación oficial, o usá un entorno Python 3.12 con `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` y `pip install streamlit plotly scikit-learn pandas pytest`.

## Qué podés hacer

| Sección | Resultado |
|---|---|
| Entrenamiento | Loss/accuracy train-val, traza por batch, Adam/AdamW/SGD, comparación de hasta 6 runs |
| Arquitectura | Bloques editables, conv individuales, BN/activación/pool, MLP, tabla de tamaños y RF, diagrama |
| Datos | Splits estratificados, normalización sólo train, preview de augmentation |
| Interpretabilidad | Filtros, feature maps, Grad-CAM, gradiente de una neurona a entrada |
| Test | Accuracy, precision/recall/F1 macro y por clase, confusión, ROC/AUC OvR, errores |
| Guía | Recorrido de 45 minutos, fórmulas y preguntas para el examen |

Leé [la guía de estudio](GUIA_ESTUDIO.md) y cambiá una variable por experimento. No ajustes sobre test.

## Reproducibilidad y artefactos

- Configuración congelada por run; los controles editados no cambian un run anterior.
- Semillas de Python/NumPy/PyTorch y algoritmo determinista CPU. La reproducibilidad exacta depende también de versión de paquetes/hardware; `uv.lock` fija versiones.
- Mejor checkpoint según loss **de validación** se usa para test/mapas.
- Exportá JSON de configuración, CSV de batches/épocas y `.pt` con pesos, normalización, clases, configuración, splits y mejor época.
- Historial en memoria de sesión: no sobrevive a reinicios. Checkpoints son para inferencia; no guardan estado del optimizador ni permiten reanudar desde UI.
- No cargues checkpoints PyTorch de fuentes desconocidas. Para cargar el exportado: `torch.load(path, map_location="cpu", weights_only=True)`; reconstruí `CNN(Config.from_dict(payload['config']), tamaño)` y aplicá `load_state_dict`.

## Verificar

```bash
uv run pytest -q
uv run python smoke.py
```

Las pruebas cubren tamaño/RF, configuraciones inválidas, splits/normalización, métricas, gradientes/hooks y AppTest del dashboard vacío, entrenamiento real y edición de controles sin reentrenamiento accidental. El smoke entrena una red real y comprueba loss finita y mejora de loss train. No sustituye validación científica exhaustiva ni garantiza un accuracy específico.

## Estructura y límites

`cnn_lab/model.py` configura y valida la CNN; `data.py` prepara splits/transformaciones; `training.py` entrena y elige checkpoint; `metrics.py` evalúa; `explain.py` produce atribuciones; `app.py` presenta el cuaderno.

Límites: 1–3 bloques, 1–3 convoluciones por bloque, canales ≤64, MLP ≤256, 30 épocas y 2M parámetros. Las operaciones son síncronas, sin controles ficticios de pausa. BatchNorm con mapas 1×1 y un último batch de una muestra se rechaza con instrucciones para cambiar batch size, muestras o arquitectura. «GEA» requiere precisar el método; se implementan Grad-CAM y sensibilidad de neurona, sin llamarlos GEA. Mapas Digits son de baja resolución y no prueban causalidad.

**Rollback:** este laboratorio es una unidad independiente; retirar `app.py`, `cnn_lab/`, tests, documentación, `pyproject.toml`, `uv.lock` y `.streamlit/` elimina su comportamiento sin afectar otros proyectos.

## Animales RGB y tus propias imágenes

1. En el sidebar elegí **Animales CIFAR-10** y cargá el preset, o elegí **CIFAR-10 · solo animales** en Datos.
2. El presupuesto inicial es **3000 imágenes oficiales de entrenamiento seleccionadas**, unas 500 por clase **antes** de reservar 20% de validación (aprox. 400 train + 100 val por clase). Test viene de otro subconjunto del test oficial. Podés usar 600–1200 para una prueba rápida.
3. Entrená. Luego abrí **Probar imagen**, subí PNG/JPEG/WebP y, opcionalmente, seleccioná la clase esperada.

Se conservan sólo los IDs CIFAR-10 **2, 3, 4, 5, 6, 7**, remapeados a **pájaro, gato, ciervo, perro, rana, caballo**. Vehículos nunca entran a train/val/test de este experimento. Las imágenes son **RGB 32×32**, la CNN recibe 3 canales y produce 6 logits. La descarga inicial contiene el archivo fuente completo de CIFAR-10 (~163 MB), necesario aunque sólo entrenemos animales; queda cacheado en `data/`. No se publica ni se envía tu imagen a servicios externos.

**Probar imagen siempre usa el último entrenamiento completado**, con su mejor checkpoint de validación, aunque estés comparando un run anterior o editando controles. Se muestra la imagen original y la entrada efectiva. Se aplica orientación EXIF, transparencia sobre blanco, conversión RGB/gris según la red, resize bilineal sin recorte, escala [0,1] y normalización congelada de train (por canal para RGB). No entrena ni usa augmentation en la subida. Límite: **10 MB y 16 MP**. Las fotos no son automáticamente recortadas al animal.

Este es un **clasificador cerrado**, no un detector de animales desconocidos: siempre elige una clase aprendida. Una foto fuera de dominio puede recibir alta confianza incorrecta; softmax no garantiza corrección. CIFAR-10 es más difícil que Digits y unas pocas épocas/subconjuntos no garantizan buena precisión.

Los nuevos checkpoints incluyen `format_version=2`, `input_shape`, clases y estadísticas RGB tensoriales `(1,C,1,1)`. Los checkpoints grises anteriores conservan sus estadísticas escalares y siguen reconstruyéndose con `CNN(Config.from_dict(payload['config']))`; para archivos v2 también podés usar los valores de `input_shape` explícitamente. La UI importa configuración JSON, **no pesos externos**.

Pruebas adicionales (mocks sin descargas) y smoke de integración explícito:

```bash
uv run pytest -q
# Descarga oficial una vez; 600 muestras seleccionadas, 3 épocas CPU:
uv run python smoke_animals.py
```

Referencias: [torchvision CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html), [dataset original](https://www.cs.toronto.edu/~kriz/cifar.html).
