"""Módulo de preprocesamiento de imágenes (capa CONTROLADOR).

Transforma una imagen RGB uint8 (salida de ``read_img``) en el tensor de
entrada que espera la CNN: ``np.float32`` con forma ``(1, 512, 512, 1)`` en
rango ``[0.0, 1.0]``. Todas las funciones son puras: ninguna modifica el
arreglo recibido como argumento.

Pipeline (orden fijo, ver ``preprocess``):
    1. ``resize_image``:  ajusta la imagen a 512x512.
    2. ``to_grayscale``:  convierte a un solo canal.
    3. ``apply_clahe``:   realza contraste local.
    4. ``normalize``:     escala a [0.0, 1.0] en float32.
    5. ``to_batch``:      agrega dimensiones de batch y canal.

Este módulo pertenece al Controlador de la arquitectura MVC: no importa
``tkinter`` ni ``tensorflow``/``keras``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_NDIM_2D = 2
_NDIM_3D = 3
_NUM_CANALES_RGB = 3


def resize_image(
    image: np.ndarray,
    size: tuple[int, int] = config.IMAGE_SIZE,
) -> np.ndarray:
    """Redimensiona una imagen a un tamaño fijo usando interpolación por área.

    Se usa ``cv2.INTER_AREA`` porque es la interpolación recomendada por
    OpenCV cuando se **reduce** el tamaño de una imagen: promedia los píxeles
    del área de origen que caen en cada píxel de destino, lo que evita el
    aliasing (patrones moiré / pérdida de detalle) que producen métodos como
    ``INTER_LINEAR`` o ``INTER_NEAREST`` al reducir. Para una radiografía,
    preservar bordes y texturas finas sin artefactos es relevante porque el
    modelo y Grad-CAM dependen de esos detalles.

    Args:
        image: Arreglo de imagen 2D (H, W) o 3D (H, W, C). No se modifica.
        size: Tupla ``(ancho, alto)`` destino. Por defecto ``config.IMAGE_SIZE``.

    Returns:
        np.ndarray: Copia redimensionada de ``image``, mismo dtype y número
        de canales que la entrada.

    Raises:
        ValueError: Si ``image`` no es un ``np.ndarray`` 2D o 3D.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (_NDIM_2D, _NDIM_3D):
        forma = getattr(image, "shape", None)
        raise ValueError(f"resize_image espera un np.ndarray 2D o 3D, recibido shape={forma}")
    origen = image.copy()
    redimensionada = cv2.resize(origen, size, interpolation=cv2.INTER_AREA)
    return redimensionada


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen RGB a escala de grises (1 canal).

    Si la imagen ya viene en 2D (escala de grises), la función es
    idempotente: devuelve una copia sin alterar los valores.

    Args:
        image: Arreglo ``uint8`` de forma ``(H, W)`` o ``(H, W, 3)`` (RGB).

    Returns:
        np.ndarray: Arreglo ``uint8`` de forma ``(H, W)``.

    Raises:
        ValueError: Si ``image`` no es ``np.ndarray``, si su dtype no es
            ``uint8``, o si su forma no es ``(H, W)`` ni ``(H, W, 3)``.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"to_grayscale espera un np.ndarray, recibido {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"to_grayscale espera dtype uint8, recibido dtype={image.dtype}")
    if image.ndim == _NDIM_2D:
        return image.copy()
    if image.ndim == _NDIM_3D and image.shape[2] == _NUM_CANALES_RGB:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"to_grayscale espera forma (H, W) o (H, W, 3), recibido shape={image.shape}")


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = config.CLAHE_TILE_GRID,
) -> np.ndarray:
    """Aplica CLAHE (ecualización adaptativa de histograma con recorte).

    CLAHE (Contrast Limited Adaptive Histogram Equalization) divide la
    imagen en bloques (``tileGridSize``) y ecualiza el histograma de cada
    bloque de forma independiente, en vez de usar un único histograma
    global como la ecualización clásica (``cv2.equalizeHist``). Esto es
    importante en radiografías de tórax porque el contraste útil (bordes de
    costillas, opacidades pulmonares) suele concentrarse en rangos de
    intensidad estrechos y localizados; una ecualización global tiende a
    sobre-amplificar el ruido de fondo y aplanar detalles clínicamente
    relevantes. El parámetro ``clip_limit`` acota la altura máxima del
    histograma de cada bloque antes de redistribuir el excedente: valores
    bajos limitan la amplificación de contraste (y de ruido); valores altos
    permiten más contraste pero también más ruido.

    Args:
        image: Arreglo ``uint8`` de forma ``(H, W)`` (un solo canal).
        clip_limit: Límite de recorte del histograma. Por defecto
            ``config.CLAHE_CLIP_LIMIT``.
        tile_grid_size: Tamaño de la cuadrícula de bloques ``(filas, cols)``.
            Por defecto ``config.CLAHE_TILE_GRID``.

    Returns:
        np.ndarray: Arreglo ``uint8`` de igual forma que ``image``, con
        contraste local realzado.

    Raises:
        ValueError: Si ``image`` no es 2D o su dtype no es ``uint8``.
    """
    if not isinstance(image, np.ndarray) or image.ndim != _NDIM_2D:
        forma = getattr(image, "shape", None)
        raise ValueError(
            f"apply_clahe espera una imagen 2D en escala de grises, recibido shape={forma}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"apply_clahe espera dtype uint8, recibido dtype={image.dtype}")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image.copy())


def normalize(image: np.ndarray) -> np.ndarray:
    """Escala una imagen a rango ``[0.0, 1.0]`` en ``float32``.

    Es robusta ante entradas ya normalizadas: si el arreglo ya es de tipo
    flotante y su valor máximo no excede 1.0, no se vuelve a dividir por
    255 (evita doble normalización).

    Args:
        image: Arreglo numérico (``uint8`` o flotante) de cualquier forma.

    Returns:
        np.ndarray: Copia en ``float32`` con valores en ``[0.0, 1.0]``.

    Raises:
        ValueError: Si ``image`` no es un ``np.ndarray``.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"normalize espera un np.ndarray, recibido {type(image)}")
    arreglo = image.astype(np.float32, copy=True)
    valor_maximo = float(arreglo.max()) if arreglo.size > 0 else 0.0
    if valor_maximo > 1.0:
        arreglo = arreglo / 255.0
    return arreglo


def to_batch(image: np.ndarray) -> np.ndarray:
    """Agrega las dimensiones de batch y canal a una imagen 2D.

    Convierte ``(H, W)`` en ``(1, H, W, 1)``, formato que espera la CNN de
    Keras (batch, alto, ancho, canales).

    Args:
        image: Arreglo 2D de forma ``(H, W)``.

    Returns:
        np.ndarray: Arreglo ``float32`` de forma ``(1, H, W, 1)``.

    Raises:
        ValueError: Si ``image`` no es 2D.
    """
    if not isinstance(image, np.ndarray) or image.ndim != _NDIM_2D:
        forma = getattr(image, "shape", None)
        raise ValueError(f"to_batch espera una imagen 2D (H, W), recibido shape={forma}")
    lote = np.expand_dims(image, axis=0)
    lote = np.expand_dims(lote, axis=-1)
    return lote.astype(np.float32, copy=True)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Ejecuta el pipeline completo de preprocesamiento sobre una imagen RGB.

    Pipeline aplicado, en orden:
        1. ``resize_image``: ajusta la imagen a ``config.IMAGE_SIZE``.
        2. ``to_grayscale``: reduce a un solo canal de intensidad.
        3. ``apply_clahe``: realza el contraste local (CLAHE).
        4. ``normalize``: escala los valores a ``[0.0, 1.0]`` en ``float32``.
        5. ``to_batch``: agrega dimensiones de batch y canal.

    Args:
        image: Arreglo ``np.ndarray`` RGB ``uint8`` de forma ``(H, W, 3)``
            (contrato de salida de ``read_img``), o ``(H, W)`` en escala de
            grises.

    Returns:
        np.ndarray: Tensor ``float32`` de forma ``(1, 512, 512, 1)`` con
        valores en ``[0.0, 1.0]``, listo para ``model.predict``.

    Raises:
        ValueError: Si ``image`` no es un ``np.ndarray`` 2D o 3D.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (_NDIM_2D, _NDIM_3D):
        forma = getattr(image, "shape", None)
        raise ValueError(f"preprocess espera un np.ndarray 2D o 3D, recibido shape={forma}")
    logger.debug("preprocess: entrada shape=%s dtype=%s", image.shape, image.dtype)

    redimensionada = resize_image(image)
    logger.debug(
        "preprocess: tras resize_image shape=%s dtype=%s",
        redimensionada.shape,
        redimensionada.dtype,
    )

    gris = to_grayscale(redimensionada)
    logger.debug("preprocess: tras to_grayscale shape=%s dtype=%s", gris.shape, gris.dtype)

    realzada = apply_clahe(gris)
    logger.debug("preprocess: tras apply_clahe shape=%s dtype=%s", realzada.shape, realzada.dtype)

    normalizada = normalize(realzada)
    logger.debug(
        "preprocess: tras normalize shape=%s dtype=%s min=%.4f max=%.4f",
        normalizada.shape,
        normalizada.dtype,
        float(normalizada.min()),
        float(normalizada.max()),
    )

    lote = to_batch(normalizada)
    logger.debug("preprocess: salida shape=%s dtype=%s", lote.shape, lote.dtype)
    return lote
