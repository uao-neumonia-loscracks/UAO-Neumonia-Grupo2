# Pipeline de preprocesamiento — `src/preprocess_img.py`

> **Advertencia:** esta herramienta es de apoyo educativo, **no** es un
> dispositivo médico certificado. No debe usarse para diagnóstico clínico real.

Este documento explica, paso a paso, cómo una imagen leída por `read_img`
(RGB `uint8`, forma `(H, W, 3)`) se transforma en el tensor de entrada de la
CNN (`float32`, forma `(1, 512, 512, 1)`, rango `[0.0, 1.0]`). Pensado como
guion para la sustentación oral.

## Tabla del pipeline

| Paso | Función | Entrada | Salida | Por qué |
|---|---|---|---|---|
| 1 | `resize_image` | `(H, W, 3)` uint8, cualquier tamaño | `(512, 512, 3)` uint8 | La CNN espera entrada de tamaño fijo. Se usa `cv2.INTER_AREA` porque, al **reducir** el tamaño de una imagen, promedia los píxeles del área de origen y evita aliasing (patrones moiré, pérdida de bordes finos) que sí introducen `INTER_LINEAR`/`INTER_NEAREST` en reducción. |
| 2 | `to_grayscale` | `(512, 512, 3)` uint8 RGB | `(512, 512)` uint8 | La CNN del proyecto fue entrenada con un solo canal de intensidad; el color no aporta información diagnóstica en una radiografía (que ya es monocroma en origen). Es idempotente: si ya llega en 2D, se devuelve una copia sin alterar. |
| 3 | `apply_clahe` | `(512, 512)` uint8 | `(512, 512)` uint8, contraste realzado | CLAHE (*Contrast Limited Adaptive Histogram Equalization*) ecualiza el histograma por bloques (`tileGridSize=(4,4)`) en vez de globalmente, realzando estructuras pulmonares locales sin saturar el resto de la imagen. `clipLimit=2.0` acota cuánto se amplifica el contraste por bloque, evitando amplificar ruido de fondo. |
| 4 | `normalize` | `(512, 512)` uint8 | `(512, 512)` float32, rango `[0.0, 1.0]` | Las redes neuronales convergen mejor y de forma más estable con entradas en un rango pequeño y acotado (`[0,1]`) en vez de `[0,255]`. |
| 5 | `to_batch` | `(512, 512)` float32 | `(1, 512, 512, 1)` float32 | Keras espera un tensor con dimensión de *batch* (aquí siempre 1, una imagen a la vez) y dimensión explícita de canal (1, escala de grises). |

## Composición: `preprocess(image)`

`preprocess` encadena los 5 pasos anteriores en ese orden exacto y registra
(`logging.debug`) la forma y tipo del arreglo después de cada paso, lo que
facilita depurar errores de forma/dtype durante la demo en vivo.



## Garantías de diseño

- **Pureza:** ninguna función modifica el arreglo recibido (se usan copias
  vía `.copy()` / `astype(copy=True)`); verificado con tests dedicados de
  pureza en `tests/test_preprocess_img.py`.
- **Validación de entradas:** cada función levanta `ValueError` con la forma
  recibida cuando el contrato no se cumple (por ejemplo, `to_grayscale` con
  dtype flotante, o `apply_clahe` con un arreglo 3D).
- **Riesgo clínico pendiente:** el mapeo de clases (`0=bacteriana,
  1=normal, 2=viral`) usado aguas abajo por el modelo aún no está verificado
  empíricamente con imágenes de clase conocida — sigue como TODO abierto del
  proyecto, no afecta a este módulo pero es crítico antes de cualquier demo
  con interpretación clínica.

  