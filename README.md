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

Límites: 1–3 bloques, 1–3 convoluciones por bloque, canales ≤64, MLP de 0–5 capas ocultas, ≤256 neuronas por capa, 30 épocas y 2M parámetros. Las operaciones son síncronas, sin controles ficticios de pausa. BatchNorm con mapas 1×1 y un último batch de una muestra se rechaza con instrucciones para cambiar batch size, muestras o arquitectura. Mapas Digits son de baja resolución y no prueban causalidad.

**Rollback:** este laboratorio es una unidad independiente; retirar `app.py`, `cnn_lab/`, tests, documentación, `pyproject.toml`, `uv.lock` y `.streamlit/` elimina su comportamiento sin afectar otros proyectos.

## Animales RGB y tus propias imágenes

1. En el sidebar elegí **Animales CIFAR-10** y cargá el preset, o elegí **CIFAR-10 · solo animales** en Datos.
2. El presupuesto inicial es **3000 imágenes oficiales de entrenamiento seleccionadas**, unas 500 por clase **antes** de reservar 20% de validación (aprox. 400 train + 100 val por clase). Test viene de otro subconjunto del test oficial. Podés usar 600–1200 para una prueba rápida.
3. Entrená. Luego abrí **Probar imagen**, subí PNG/JPEG/WebP y, opcionalmente, seleccioná la clase esperada.

Se conservan sólo los IDs CIFAR-10 **2, 3, 4, 5, 6, 7**, remapeados a **pájaro, gato, ciervo, perro, rana, caballo**. Vehículos nunca entran a train/val/test de este experimento. Las imágenes son **RGB 32×32**, la CNN recibe 3 canales y produce 6 logits. La descarga inicial contiene el archivo fuente completo de CIFAR-10 (~163 MB), necesario aunque sólo entrenemos animales; queda cacheado en `data/`. No se publica ni se envía tu imagen a servicios externos.

**Probar imagen siempre usa el último entrenamiento completado**, con su mejor checkpoint de validación, aunque estés comparando un run anterior o editando controles. Se muestra la imagen original y la entrada efectiva. Se aplica orientación EXIF, transparencia sobre blanco, conversión RGB/gris según la red, resize bilineal sin recorte, escala [0,1] y normalización congelada de train (por canal para RGB). No entrena ni usa augmentation en la subida. Límite: **10 MB y 16 MP**. Las fotos no son automáticamente recortadas al animal.

Este es un **clasificador cerrado**, no un detector de animales desconocidos: siempre elige una clase aprendida. Una foto fuera de dominio puede recibir alta confianza incorrecta; softmax no garantiza corrección. CIFAR-10 es más difícil que Digits y unas pocas épocas/subconjuntos no garantizan buena precisión.

Los nuevos checkpoints incluyen `format_version=4`, `input_shape`, clases y estadísticas RGB tensoriales `(1,C,1,1)`. Los checkpoints grises anteriores conservan sus estadísticas escalares y siguen reconstruyéndose con `CNN(Config.from_dict(payload['config']))`; para archivos v2/v3 también podés usar los valores de `input_shape` explícitamente. La UI importa configuración JSON, **no pesos externos**.

Pruebas adicionales (mocks sin descargas) y smoke de integración explícito:

```bash
uv run pytest -q
# Descarga oficial una vez; 600 muestras seleccionadas, 3 épocas CPU:
uv run python smoke_animals.py
```

Referencias: [torchvision CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html), [dataset original](https://www.cs.toronto.edu/~kriz/cifar.html).


## Capas densas: la MLP también tiene ocultas

En **Arquitectura**, elegí la profundidad **0–5 capas densas ocultas** y configurá cada una: neuronas, activación, dropout y bias. La capa densa de salida se agrega siempre aparte: **6 logits** para animales o **10 logits** para los otros datasets. No agrega softmax antes de CrossEntropyLoss.

Ejemplo animales: `Flatten/GAP → Dense128/ReLU/dropout0.2 → Dense64/GELU/dropout0.1 → Dense6 (logits)`. Son **2 capas ocultas + 1 capa de salida**, no una MLP sin ocultas. Profundidad 0: sólo la salida lineal sobre las características de la CNN. La tabla y diagrama muestran cada operación y la entrada/salida de las densas.

Configuración nueva canónica: `dense: [{"units":128,"activation":"relu","dropout":0.2,"bias":true}, ...]`. JSON/checkpoints antiguos con `hidden`, `activation` y `dropout` globales se migran automáticamente a capas individuales conservando sus valores. Los nuevos exports no mantienen esos campos redundantes. Checkpoint v4 conserva orden/nombres de parámetros de la base, estadísticas y shape; v1/v2 siguen reconstruibles mediante `Config.from_dict`.

**Presets de configuración** son recetas, no modelos preentrenados: cargar reemplaza arquitectura, hiperparámetros y dataset del editor, no modelos/resultados anteriores. Inicializamos pesos aleatorios sólo al entrenar. El expander del sidebar explica exactamente cada preset.

## Validación K-fold sin tocar test

En **Entrenamiento → Validación K-fold estratificada** elegí K=2/3/5 y épocas por fold. El botón inicia **K entrenamientos independientes**, así que cuesta aproximadamente K veces un entrenamiento. No se ejecuta automáticamente.

- Usa solamente el pool de desarrollo: train + validación originales. Test conserva sus índices y **no se predice ni se calcula ninguna métrica test** en esta acción.
- Cada muestra de desarrollo pertenece a validación una vez; las proporciones de clases se conservan aproximadamente.
- Cada fold reinicia modelo/optimizador y tiene semilla registrada. Su media/desvío se calculan sólo con train de ese fold; augmentation sólo allí.
- Muestra métricas macro por fold, mejor loss/época de validación, curvas/trazas y media/desvío entre folds (**ddof=1**, no intervalo de confianza). Descargá CSV o JSON con configuración, índices y logs.
- Se conserva sólo el último resultado K-fold y ningún modelo de fold; no reemplaza la última CNN para **Probar imagen**. Tras comparar, entrená normalmente la configuración elegida.

Se elige checkpoint con validación del mismo fold: es un procedimiento práctico de estudio, **no nested CV** ni una estimación totalmente libre de sesgo de selección. No uses test para elegir K, épocas o arquitectura.

```bash
# Tres folds × dos épocas, Digits offline; sin evaluar test:
uv run python smoke_kfold.py
```

Referencias: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [validación cruzada](https://scikit-learn.org/stable/modules/cross_validation.html).

## Arquitectura interactiva: capas y neuronas

En **Arquitectura → Vista del diagrama** hay tres vistas Plotly locales:

- **Bloques CNN + MLP:** esquema vertical legible de entrada, bloques convolucionales, Flatten/GAP, densas ocultas y logits; formas y parámetros reales.
- **Operaciones detalladas:** todas las convoluciones, BN, activaciones, pooling y operaciones de la MLP. El panel vertical se desplaza sin reducir tipografía. Hover muestra kernel/stride/padding, formas, parámetros y RF cuando tiene sentido.
- **Neuronas de la MLP:** puntos por unidad y conexiones densas. Hasta **12 unidades representativas** por vector/oculta, con índices reales y total explícito; todas las clases de salida. Las neuronas omitidas también existen y tienen conexiones. Activación/dropout se anotan, no agregan neuronas.

Arrastrá para mover, rueda para zoom y barra del gráfico para restaurar vista/descargar PNG. Los nodos de bloques representan operaciones, no cada neurona. La CNN se resume en un vector en la vista MLP: **no se afirma que las convoluciones sean fully connected**. Son esquemas funcionales, no ubicación física de memoria/GPU. No requieren servicios ni scripts CDN adicionales.

## Parada temprana: opcional, desactivada por defecto

En **Entrenamiento → Parada temprana** podés activar la regla y definir **paciencia** (1–30 épocas) y **mejora mínima** (0–10 unidades de loss). Se monitorea exclusivamente la loss de validación al final de cada época. La primera fija la referencia; una reducción **estrictamente mayor** que `min_delta` reinicia el contador. Tras `patience` épocas consecutivas sin esa mejora, termina. Igualdad no cuenta como mejora.

**Detenerse no es lo mismo que guardar el checkpoint:** los pesos restaurados siempre corresponden al **mínimo absoluto de loss de validación**, aunque esa mejora pequeña no alcance para reiniciar paciencia. Se informa cuántas épocas se completaron del máximo y el motivo; las curvas señalan checkpoint/parada. Si la paciencia se agota justo en la última época, no se afirma que se ahorraron épocas.

La misma opción se aplica a cada fold con un contador nuevo. K-fold sigue sin evaluar test ni reemplazar la última CNN de inferencia. CSV de épocas incluye contador/referencia; JSON/config y checkpoints v4 guardan `epochs_completed`, `max_epochs`, `stopped_early` y `stop_reason`. Configuraciones anteriores importadas quedan con parada desactivada.

```bash
uv run python smoke_early_stopping.py
```

El smoke usa `min_delta=10` deliberadamente para probar la mecánica (3 de 8 épocas con paciencia2). **No demuestra que hubiera sobreajuste natural.** Para experimentar de verdad, empezá con `min_delta=0.0001`, paciencia3 y compará curvas sin ajustar sobre test.

## Controles de bloques sin ambigüedad

- **Cantidad de filtros / mapas de salida:**16 filtros generan16 canales, no un kernel de lado16. Cada filtro completo abarca todos los canales de entrada.
- **Lado del filtro · k×k:**3 significa un filtro espacial3×3. Campo editable entero1–7, incluidos pares; no se soportan filtros rectangulares. Padding0–3, stride1–2 y dilatación1–2.
- **Después de cada convolución:**BatchNorm opcional y activación de mapas.
- **Reducción al finalizar el bloque:**pooling con ventana cuadrada2–4; stride=lado de ventana, sin padding. Mantiene canales y reduce alto/ancho. La tabla muestra las dimensiones reales y detecta colapsos.

La interfaz usa secciones uniformes, learning rate con cinco decimales y resumen/acción principal visible; optimización, regularización y parada no quedan mezcladas. A390px los parámetros se apilan, las pestañas pueden desplazarse horizontalmente y los gráficos/tablas mantienen su propia navegación. Fuentes locales, foco de teclado visible y reducción de movimiento: sin recursos visuales externos.

## Vista dedicada: Entrenamiento en vivo

**Entrenar CNN** cambia inmediatamente del editor a un monitor y congela la configuración actual. Primero muestra preparación/caché/descarga; después inicialización, batches, validación y test. El trabajo pendiente se consume una sola vez: una recarga no inicia otro entrenamiento. Los controles no se muestran durante la ejecución. Al finalizar/error aparece **Volver a configuración**; **Ver último entrenamiento en vivo** recupera el monitor sin entrenar.

Se muestran época/máximo, batch/total, batches acumulados, muestras vistas, loss del batch y medias actuales de train, tiempo y última validación disponible. La curva por época compara loss **train y validación** y actualiza tras cada validación; la traza por batch es otra gráfica. Antes de primera validación no se inventa un valor cero. Las actualizaciones de batch se limitan aproximadamente a cada250ms; los eventos de fase/época se muestran siempre. Un fallo nunca reemplaza la última CNN completada para inferencia.

**Preparar y visualizar datos** utiliza la misma fuente de eventos: progreso de bytes durante descargas y muestras sólo tras preparar datos reales. La última preparación completada conserva su nombre de dataset si una preparación posterior falla.

## Descargas que no esperan indefinidamente

El transporte administrado usa URL y MD5 de metadatos oficiales torchvision, verifica antes de extraer y carga con `download=False`. Límites:10s de conexión,20s por lectura, unos300s por archivo (más una lectura en curso). Si Content-Length existe hay progreso de bytes; si no, se muestra tamaño recibido sin porcentaje ficticio. No hay reintentos automáticos ni resumes no verificados.

- Cache CIFAR existente: torchvision verifica archivos oficiales. Archivo nuevo: MD5 obligatorio antes de extracción segura.
- Descarga nueva utiliza `.part` privado y bloqueo entre sesiones; un parcial no se acepta como caché.
- Un archivo incompleto **reciente** del descargador antiguo no se sobrescribe: puede pertenecer a una descarga aún activa. Esperá que termine; si dejó de cambiar, reintentá después60s. La nueva implementación no puede imponer timeout retroactivo a una llamada antigua ya bloqueada.
- Alternativa manual CIFAR: descargá el [archivo oficial](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) y colocalo como `data/cifar-10-python.tar.gz`. Debe coincidir con MD5`c58f30108f718f92721af3b95e74349a`; el laboratorio lo verifica y extrae. Para MNIST/Fashion se usan los archivos/resources y primeras fuentes oficiales de torchvision; un error de esa fuente se informa, no se oculta con mirrors arbitrarios.

Una conexión externa lenta sigue siendo lenta: la mejora aporta feedback/límites/cache seguro, no promete acelerar el servidor de origen. No cierres procesos ajenos para sustituir una descarga en curso.

`tests/live_browser_fixture.py` es **sólo una prueba**: servidor aislado8502 con demoras de preparación/validación para fotografiar estados intermedios. El comando normal sigue siendo `uv run streamlit run app.py`; producción no agrega sleeps.
