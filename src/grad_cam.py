"""Módulo Grad-CAM (capa MODELO) — UAO-Neumonia.

Implementa el algoritmo Grad-CAM (Gradient-weighted Class Activation Mapping)
para explicar visualmente las predicciones de la CNN de clasificación de
neumonía sobre radiografías de tórax.

Referencia bibliográfica:
    Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
    Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization. Proceedings of the IEEE International
    Conference on Computer Vision (ICCV), 618-626.
    https://doi.org/10.1109/ICCV.2017.74

Capa MVC: MODELO. Junto con ``load_model.py``, es el único lugar del
proyecto donde vive TensorFlow/Keras. No importa ``tkinter`` ni ningún
módulo de la Vista.

Advertencia clínica:
    Esta herramienta es de apoyo EDUCATIVO. NO es un dispositivo médico
    certificado y no debe usarse para tomar decisiones clínicas reales.
"""

from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np

from load_model import get_last_conv_layer_name, load_cnn_model
from preprocess_img import preprocess

# `tf` y `keras` se obtienen de sys.modules en vez de importarlos con
# sentencias `import`/`from ... import`. Motivos:
# 1) El contrato 5.4 de load_model.py NO expone un símbolo público `keras`
#    (solo load_cnn_model, get_last_conv_layer_name, compute_sha256,
#    verify_model_integrity, model_summary_text, ModelNotFoundError). Un
#    `from load_model import keras` revienta con ImportError en cuanto se
#    ejecuta la cadena real de imports (detectado en H6 al importar
#    integrator -> grad_cam con TensorFlow real, sin mocks).
# 2) Al importar `load_model` arriba, su código de módulo ya configuró
#    TF_USE_LEGACY_KERAS y el logger de TensorFlow ANTES de ejecutar su
#    propio `import tensorflow as tf` (ver HANDOFF H4, D4.1/D4.2). Agregar
#    aquí un `import tensorflow as tf` adicional haría que `ruff --fix`
#    lo reordenara junto a cv2/numpy, ejecutándolo ANTES del import de
#    `load_model` y rompiendo ese guard.
# `sys.modules.get("tf_keras", tf.keras)` prioriza el paquete `tf_keras`
# real si ya quedó cargado (caso Plan B / TF_USE_LEGACY_KERAS=1) y cae a
# `tf.keras` en cualquier otro caso.
tf = sys.modules["tensorflow"]
keras = sys.modules.get("tf_keras", tf.keras)

logger = logging.getLogger(__name__)

IMAGE_SIZE: tuple[int, int] = (512, 512)
_EPSILON: float = 1e-8
_PORCENTAJE: float = 100.0
_ESCALA_UINT8: int = 255


def compute_gradcam_heatmap(
    batch: np.ndarray,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> tuple[np.ndarray, int, float]:
    """Calcula el mapa de calor Grad-CAM para un batch preprocesado.

    Implementa el algoritmo de Selvaraju et al. (2017) paso a paso.

    Args:
        batch: Tensor de entrada con shape (1, 512, 512, 1), dtype float32,
            rango [0, 1] (contrato de ``preprocess_img.preprocess``).
        class_index: Índice de la clase objetivo. Si es ``None``, se usa la
            clase con mayor probabilidad predicha (argmax).
        layer_name: Nombre de la capa convolucional objetivo. Si es ``None``,
            se usa ``get_last_conv_layer_name(modelo)``.

    Returns:
        Tupla ``(heatmap, class_index, probabilidad)``: ``heatmap`` es un
        ``np.ndarray`` float32 (512, 512) en [0, 1], ``class_index`` es el
        índice de clase (int) usado y ``probabilidad`` es un float en
        [0, 100].

    Raises:
        RuntimeError: Si TensorFlow no puede calcular gradientes respecto a
            la capa convolucional objetivo (grafo desconectado), o si el
            fallback no puede aplicarse porque la capa objetivo no es de
            primer nivel del modelo.
    """
    inicio = time.perf_counter()
    modelo = load_cnn_model()
    nombre_capa = layer_name or get_last_conv_layer_name(modelo)
    capa_objetivo = modelo.get_layer(nombre_capa)

    # Paso 1: construir el submodelo que expone (activaciones_conv, salida).
    # En Keras 3 los modelos Sequential/anidados a veces no permiten construir
    # un `keras.Model` funcional apoyado en `capa.output` porque el grafo no
    # queda "conectado" de forma explícita (AttributeError/ValueError). Si
    # eso ocurre, se activa el FALLBACK: reconstrucción manual del forward
    # pass capa por capa dentro del propio `GradientTape`.
    usa_fallback = False
    grad_model: keras.Model | None = None
    try:
        grad_model = keras.Model(
            inputs=modelo.inputs,
            outputs=[capa_objetivo.output, modelo.output],
        )
    except (ValueError, AttributeError, TypeError) as excepcion:
        logger.warning(
            "No se pudo construir grad_model funcional (%s); usando "
            "fallback de reconstrucción manual capa por capa.",
            excepcion,
        )
        usa_fallback = True

    entrada = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        if not usa_fallback:
            # Paso 2a: forward pass directo vía el submodelo funcional.
            activaciones_conv, predicciones = grad_model(entrada, training=False)
        else:
            # Paso 2b (FALLBACK): forward pass manual capa por capa,
            # observando explícitamente el tensor de activaciones conv.
            try:
                indice_capa = modelo.layers.index(capa_objetivo)
            except ValueError as excepcion:
                raise RuntimeError(
                    "El fallback de Grad-CAM requiere que la capa objetivo "
                    f"'{nombre_capa}' sea una capa de primer nivel del "
                    "modelo; no se soportan submodelos anidados en este "
                    "modo."
                ) from excepcion

            capas_cabeza = modelo.layers[: indice_capa + 1]
            capas_cola = modelo.layers[indice_capa + 1 :]

            x = entrada
            for capa in capas_cabeza:
                x = capa(x, training=False)
            activaciones_conv = x
            tape.watch(activaciones_conv)

            for capa in capas_cola:
                x = capa(x, training=False)
            predicciones = x

        # Paso 2c: elegir la clase objetivo (argmax si no se especifica) y
        # extraer el score escalar de esa clase.
        if class_index is None:
            indice_clase = int(tf.argmax(predicciones[0]).numpy())
        else:
            indice_clase = int(class_index)
        score = predicciones[:, indice_clase]

    # Paso 3: gradiente del score respecto a las activaciones convolucionales.
    grads = tape.gradient(score, activaciones_conv)
    if grads is None:
        raise RuntimeError(
            "Gradiente None: el grafo entre la capa convolucional objetivo "
            f"'{nombre_capa}' y la salida está desconectado."
        )

    # Paso 4: importancia por canal = promedio del gradiente sobre el batch
    # y las dimensiones espaciales (axis 0=batch, 1=alto, 2=ancho).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Paso 5: combinación ponderada de los mapas de activación por canal.
    activaciones_conv = activaciones_conv[0]
    heatmap = tf.reduce_sum(activaciones_conv * pooled_grads, axis=-1)

    # Paso 6: ReLU (solo activaciones que aumentan el score) y normalización
    # a [0, 1] con guardia anti-división-por-cero.
    heatmap = tf.maximum(heatmap, 0.0)
    valor_maximo = tf.reduce_max(heatmap)
    heatmap = heatmap / (valor_maximo + _EPSILON)

    # Paso 7: redimensionar a la resolución canónica del proyecto (512x512).
    heatmap_np = heatmap.numpy().astype(np.float32)
    heatmap_redimensionado = cv2.resize(
        heatmap_np, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    heatmap_redimensionado = np.clip(heatmap_redimensionado, 0.0, 1.0)

    probabilidad = float(predicciones.numpy()[0, indice_clase]) * _PORCENTAJE
    duracion = time.perf_counter() - inicio
    logger.info(
        "Grad-CAM calculado en %.3f s (capa=%s, clase=%d, prob=%.2f%%, fallback=%s).",
        duracion,
        nombre_capa,
        indice_clase,
        probabilidad,
        usa_fallback,
    )

    return heatmap_redimensionado, indice_clase, probabilidad


def overlay_heatmap(
    heatmap: np.ndarray,
    original_rgb: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Superpone un mapa de calor Grad-CAM sobre la radiografía original.

    Args:
        heatmap: Mapa de calor float32 (512, 512) en rango [0, 1].
        original_rgb: Imagen original RGB uint8 (H, W, 3). Se redimensiona a
            512x512 si no coincide con el contrato.
        alpha: Peso del mapa de calor en la mezcla (0=solo original,
            1=solo mapa de calor). Por defecto 0.4.

    Returns:
        Imagen RGB uint8 (512, 512, 3) con el mapa de calor superpuesto.
    """
    # Paso 1: garantizar que la imagen original tenga la resolución canónica.
    if original_rgb.shape[:2] != IMAGE_SIZE:
        original_redimensionado = cv2.resize(
            original_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
        )
    else:
        original_redimensionado = original_rgb
    original_redimensionado = original_redimensionado.astype(np.uint8)

    # Paso 2: escalar el heatmap [0,1] a uint8 [0,255] para poder colorearlo.
    heatmap_uint8 = np.uint8(_ESCALA_UINT8 * np.clip(heatmap, 0.0, 1.0))

    # Paso 3: aplicar el mapa de color JET.
    mapa_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Paso 4 (GOTCHA clásico): `cv2.applyColorMap` devuelve el color en BGR,
    # no en RGB. Si se mezcla directamente con `original_rgb` (que sí está
    # en RGB) sin convertir, los canales rojo y azul quedan intercambiados:
    # las zonas de mayor activación se verían azules en vez de rojas,
    # invirtiendo la interpretación clínica del mapa de calor.
    mapa_color_rgb = cv2.cvtColor(mapa_color_bgr, cv2.COLOR_BGR2RGB)

    # Paso 5: mezcla ponderada overlay + original.
    overlay = cv2.addWeighted(mapa_color_rgb, alpha, original_redimensionado, 1.0 - alpha, 0)

    return overlay.astype(np.uint8)


def grad_cam(image_rgb: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Fachada Grad-CAM: imagen cruda RGB -> overlay + predicción.

    Orquesta el pipeline completo: preprocesamiento (Controlador) -> cálculo
    del mapa de calor -> superposición sobre la imagen original.

    Args:
        image_rgb: Imagen RGB uint8 (H, W, 3), típicamente la salida de
            ``read_img.read_image_file``.

    Returns:
        Tupla ``(overlay, class_index, probabilidad)`` con ``overlay`` RGB
        uint8 (512, 512, 3), ``class_index`` int y ``probabilidad`` float
        en [0, 100].
    """
    batch = preprocess(image_rgb)
    heatmap, indice_clase, probabilidad = compute_gradcam_heatmap(batch)
    overlay = overlay_heatmap(heatmap, image_rgb)
    return overlay, indice_clase, probabilidad
