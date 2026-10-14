# De la teoría a una predicción

**Método de estudio:** formulá una hipótesis, cambiá una sola variable, entrená y explicá el resultado. Empezá con Digits y la CNN base. No necesitás GPU ni internet.

## Recorrido de 45 minutos

1. **10 min · Arquitectura:** mirá la tabla de tamaños. Predecí qué ocurre al quitar pooling. Entrená ambas variantes y compará parámetros y validación.
2. **10 min · Aprendizaje:** aumentá learning rate; observá ruido por batch y tendencia por época. Volvé al valor base si la loss diverge.
3. **10 min · Generalización:** preset Sobreajuste. Buscá loss train que baja mientras val sube. Añadí dropout y compará.
4. **10 min · Interpretabilidad:** seleccioná un error y su clase real como objetivo Grad-CAM. Cambiá capa. Contrastá RF teórico y gradiente de neurona.
5. **5 min · Test:** explicá un falso positivo y un falso negativo usando una fila/columna de la matriz. No ajustes ahora sobre test.

## Cómo atraviesa la red una imagen

`imagen → [conv → BN opcional → activación] × N → pooling → ... → flatten/GAP → MLP → logits (uno por clase)`

- **Convolución:** filtros compartidos detectan patrones locales. PyTorch implementa correlación cruzada (sin invertir kernel), habitualmente llamada convolución en CNN.
- **Canales de salida:** cantidad de filtros y mapas. Cada filtro abarca **todos** los canales de entrada.
- **Activación:** introduce no linealidad. Varias capas lineales sin activación siguen siendo una transformación lineal (más sesgo).
- **Pooling:** reduce dimensiones sin parámetros aprendidos. Max conserva respuestas fuertes; average resume. Reduce precisión espacial.
- **Flatten:** conserva posiciones, pero puede aumentar mucho los parámetros de la MLP.
- **GAP:** promedia cada mapa espacial, entrega un valor por canal y reduce parámetros; pierde detalle posicional.
- **Dropout:** apaga unidades aleatoriamente durante train; se desactiva en eval.

### Fórmulas para verificar con la tabla

Para una dimensión espacial:

`H_out = floor((H_in + 2P − D(K−1) − 1) / S + 1)`

Parámetros Conv2D con sesgo: `C_out × (C_in × K² + 1)`.

Parámetros Linear con sesgo: `N_in × N_out + N_out`.

BatchNorm tiene `2 × C` parámetros entrenables (gamma y beta), más estadísticas móviles no entrenables.

Campo receptivo: iniciar `r=1, j=1`. Por capa: `r_nuevo = r + (K−1) × D × j`; `j_nuevo = j × S`. Pooling usa la misma regla con D=1. El RF teórico puede incluir padding o superar la imagen; no garantiza que cada píxel influya igual. El gradiente local revela sensibilidad en una entrada concreta, y puede ser cero por saturación o caminos inactivos.

## Loss, logits y optimizadores

**Logits** son scores reales no normalizados. `CrossEntropyLoss` recibe logits y etiquetas enteras; combina log-softmax y pérdida logarítmica de forma estable. **No pongas softmax antes.** Para probabilidades en test/inferencia sí aplicamos softmax.

Loss mide calidad/confianza de la distribución; accuracy sólo si la clase de mayor score fue correcta. Puede mejorar loss sin cambiar accuracy. Puede bajar accuracy con unos pocos ejemplos mientras mejora el promedio de loss.

| Optimizador | Idea | Experimento |
|---|---|---|
| SGD | Paso proporcional al gradiente | Comparar lr 0.01 vs 0.05 |
| SGD + momentum | Acumula dirección y suaviza oscilaciones | Momentum 0 vs 0.9 |
| Adam | Primer/segundo momentos, paso adaptado por parámetro | Base lr 0.003 |
| AdamW | Separa weight decay de la actualización adaptativa | Añadir decay 0.01 |

La pérdida train del dashboard se acumula mientras los pesos cambian. Validación se calcula al final con un modelo fijo en eval. Por eso no son exactamente comparables batch a batch. Elegimos el checkpoint con **menor loss de validación**, no el último ni el de mejor test.

## Normalización ≠ BatchNorm ≠ augmentation

- **Normalización de entrada:** `(x − media_train) / desvío_train`. Estadísticas calculadas sólo con train. Son fijas para val/test.
- **BatchNorm:** normaliza activaciones internas durante train con estadísticas de batch y aprende escala/desplazamiento. En eval utiliza estadísticas móviles. Batch pequeño puede hacerlas ruidosas.
- **Augmentation:** crea variaciones de train; no genera nuevas etiquetas. Se aplica rotación/traslación pequeña antes de normalizar. No espejamos dígitos: la transformación puede cambiar la identidad o producir ejemplos no realistas.

**Experimento:** base vs augmentation, luego base vs BN. Mantené semilla y splits. Un peor resultado aislado no demuestra que una técnica sea inútil: probá varias semillas y explicá el dominio.

## Métricas: qué pregunta responde cada una

La matriz muestra **filas = clase real; columnas = predicción**. Para una clase (one-vs-rest):

- `Precision = TP/(TP+FP)`: de los que predije positivos, ¿cuántos eran positivos?
- `Recall = TP/(TP+FN)`: de los positivos reales, ¿cuántos encontré?
- `F1 = 2PR/(P+R)`: equilibrio armónico; penaliza una de las dos muy baja.
- `Accuracy = aciertos/total`: puede ocultar problemas con clases desbalanceadas.
- **Macro:** promedio simple por clase, cada clase pesa igual. **Weighted:** pondera por soporte; no lo confundas con macro.
- **ROC:** TPR vs FPR variando umbral sobre scores/probabilidades. Multiclase: cada clase contra las restantes.
- **AUC:** área ROC; mide ordenamiento/discriminación, no calibración ni accuracy a un umbral. Macro AUC aquí promedia AUC de clases definidas.

Divisiones sin positivos predichos se reportan como 0 en precision/F1. ROC es indefinida si no hay positivos o no hay negativos; no se fabrica una curva. La confianza softmax no es garantía de certeza ni de calibración.

## Interpretabilidad sin promesas falsas

**Filtros** son pesos aprendidos. **Mapas** son respuestas a esta entrada. **Grad-CAM** pondera mapas por gradientes del logit objetivo y conserva contribuciones positivas. **Gradiente de neurona** mide la sensibilidad absoluta de una unidad convolucional preactivación a cada píxel. No son la misma técnica ni una explicación causal.

«GEA» no identifica aquí un método inequívoco. Este laboratorio implementa los dos métodos anteriores sin atribuirles ese nombre. Los mapas se normalizan para visualización; un mapa constante/cero se informa explícitamente. En Digits, 8×8 y dos poolings producen mapas muy pequeños: MNIST/Fashion permiten ver más detalle.

## Preguntas para el examen

1. ¿Por qué compartir pesos reduce parámetros respecto de una MLP sobre píxeles?
2. ¿Qué cambia al usar stride 2 frente a pooling 2?
3. ¿Cómo puede un RF teórico ser mayor que la imagen?
4. ¿Por qué cambiar BN a modo eval importa para test?
5. ¿Por qué elegir hiperparámetros con test contamina la evaluación?
6. ¿Por qué softmax no debe preceder a CrossEntropyLoss en este código?
7. ¿Un mapa de calor brillante demuestra causalidad? ¿Qué prueba realmente?

## Límites y próximos experimentos

Todo corre en CPU y se limita a 3 bloques, 64 canales por conv, 2 capas ocultas, 30 épocas y 2 millones de parámetros. El entrenamiento es síncrono, sin pausa/reanudación. El historial vive en la sesión; descargá CSV/JSON/checkpoint. Para estudiar mañana priorizá hipótesis y comparación controlada antes que una red grande.

## Fuentes para profundizar

- [Digits: documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- [Fashion-MNIST: repositorio oficial y clases](https://github.com/zalandoresearch/fashion-mnist).
- [Grad-CAM: artículo original](https://arxiv.org/abs/1610.02391).
- [Understanding the Effective Receptive Field in Deep CNNs](https://arxiv.org/abs/1701.04128): contexto teórico; el visualizador aquí es un gradiente local de una neurona, no una reproducción del protocolo completo del artículo.

## Experimento RGB: animales y una foto nueva

El preset **Animales CIFAR-10** usa RGB 32×32 y 6 clases: pájaro, gato, ciervo, perro, rana y caballo. Se filtran las otras 4 clases **antes** de muestrear/dividir. Las 3000 muestras iniciales equivalen aproximadamente a 500 por clase antes de validación, no 500 train reales por clase: luego se reservan 20% para val. El test procede del test oficial separado. La descarga inicial completa ronda 163 MB, aunque los vehículos se excluyen de todas las particiones utilizadas.

**Pregunta:** con el mismo kernel y canales de salida, ¿cuántos parámetros suma pasar de 1 a 3 canales de entrada? Mirá el primer bloque: cada filtro contiene un kernel por canal y suma sus contribuciones. La normalización RGB aprende **una media y un desvío por canal usando sólo train**.

1. Entrená primero 600–1200 muestras para comprobar el flujo; después probá 3000 si tenés tiempo. No esperes la precisión de Digits: hay fondos, poses y clases visualmente parecidas.
2. En **Probar imagen**, subí una foto de una de las seis clases. Observá el resize 32×32 real: ¿el animal sigue reconocible?
3. Indicá la clase esperada y comprobá la predicción. Cambiá fondo/encuadre y pensá cómo afecta el cambio de dominio.
4. Probá una imagen ajena. La red igualmente elige una clase: **no tiene una clase «desconocido» ni un detector de fuera de distribución**. Alta softmax puede ser un error muy confiado, no certeza.

La clasificación usa el **último entrenamiento completado**, no el antiguo seleccionado para comparación ni la configuración actualmente editada. Usa su checkpoint elegido con validación y su preprocesado congelado. No hay aprendizaje con las imágenes subidas. Orientación EXIF y transparencia se resuelven antes de convertir/redimensionar; las imágenes no se recortan automáticamente al animal.

[Dataset oficial CIFAR-10 y lista de clases](https://www.cs.toronto.edu/~kriz/cifar.html) · [torchvision CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
