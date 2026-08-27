"""Módulo controlador para lectura de imágenes médicas (DICOM y formatos estándar).

Pertenece a la capa Controlador de la arquitectura MVC del proyecto UAO-Neumonia.
Lee archivos DICOM o de imagen estándar (JPG, PNG, etc.) desde disco y los
convierte a un formato estandarizado: un arreglo NumPy RGB de 8 bits y una
imagen PIL equivalente, listos para ser consumidos por ``preprocess_img.py``.

Este módulo NO importa Tkinter ni ningún componente de la Vista, respetando la
separación estricta de responsabilidades del proyecto (regla de dependencia:
Controlador -> Modelo, nunca Controlador -> Vista).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError

from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

DICOM_EXTENSIONS: frozenset[str] = frozenset({".dcm", ".dicom"})
"""Extensiones de archivo reconocidas como DICOM y enrutadas a ``read_dicom_file``."""


class UnsupportedFormatError(Exception):
    """Se lanza cuando la extensión del archivo no está soportada.

    Indica que la extensión del archivo solicitado no figura en
    ``config.SUPPORTED_EXTENSIONS``, por lo que no existe un lector capaz de
    procesarlo.
    """


class ImageReadError(Exception):
    """Se lanza cuando un archivo con extensión soportada no pudo leerse.

    Cubre casos como archivos DICOM corruptos, sin ``PixelData`` válido, o
    archivos de imagen estándar dañados o ilegibles por PIL.
    """


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normaliza un arreglo numérico al rango uint8 [0, 255] usando su rango real.

    A diferencia de una normalización ingenua (dividir por 65535), esta función
    usa el mínimo y máximo reales del arreglo de entrada, lo cual es necesario
    porque los datos DICOM pueden venir en 12, 16 bits o rangos arbitrarios tras
    aplicar ``RescaleSlope``/``RescaleIntercept``.

    Args:
        array: Arreglo de entrada con cualquier dtype numérico (p. ej. int16,
            uint16, float64) y rango arbitrario de valores.

    Returns:
        np.ndarray: Arreglo de dtype ``uint8`` con los valores reescalados
        linealmente entre el mínimo y el máximo reales del arreglo de entrada.

    Example:
        >>> arreglo = np.array([100, 4095], dtype=np.uint16)
        >>> _normalize_to_uint8(arreglo)
        array([  0, 255], dtype=uint8)
    """
    arreglo_float = array.astype(np.float64)
    minimo = arreglo_float.min()
    maximo = arreglo_float.max()
    rango = maximo - minimo
    if rango == 0:
        logger.warning(
            "El arreglo de píxeles tiene rango dinámico nulo (valor constante=%s); "
            "se devuelve un arreglo plano.",
            minimo,
        )
        return np.zeros_like(arreglo_float, dtype=np.uint8)
    normalizado = (arreglo_float - minimo) / rango * 255.0
    return np.clip(normalizado, 0, 255).astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replica un canal de escala de grises en 3 canales para formar una imagen RGB.

    Args:
        gray: Arreglo 2D de forma (H, W) y dtype ``uint8``.

    Returns:
        np.ndarray: Arreglo 3D de forma (H, W, 3) y dtype ``uint8``, con los
        tres canales idénticos entre sí.

    Example:
        >>> gris = np.zeros((2, 2), dtype=np.uint8)
        >>> _gray_to_rgb(gris).shape
        (2, 2, 3)
    """
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def read_dicom_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Lee un archivo DICOM y lo convierte a un arreglo RGB de 8 bits.

    Aplica ``RescaleSlope``/``RescaleIntercept`` si están presentes en el
    dataset, invierte los valores de intensidad cuando
    ``PhotometricInterpretation`` es ``MONOCHROME1`` (donde el valor mínimo se
    muestra como blanco), y normaliza el resultado al rango uint8 [0, 255]
    usando el rango dinámico real del arreglo de píxeles (soporta 12, 16 bits
    o cualquier profundidad, sin asumir 65535 como máximo).

    Args:
        path: Ruta al archivo DICOM.

    Returns:
        tuple[np.ndarray, PIL.Image.Image]: Una tupla con el arreglo RGB
        ``uint8`` de forma (H, W, 3) y la imagen PIL equivalente en modo
        "RGB".

    Raises:
        FileNotFoundError: Si ``path`` no existe en el sistema de archivos.
        ImageReadError: Si el archivo no es un DICOM válido, o carece de
            ``PixelData`` legible.

    Example:
        >>> arreglo, imagen = read_dicom_file("radiografia.dcm")
        >>> arreglo.shape
        (512, 512, 3)
    """
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo DICOM no existe: {ruta}")

    logger.info("Leyendo archivo DICOM: %s", ruta)
    try:
        dataset = pydicom.dcmread(str(ruta))
    except (InvalidDicomError, OSError, ValueError) as exc:
        raise ImageReadError(f"No se pudo leer el DICOM '{ruta}': {exc}") from exc

    try:
        pixel_array = dataset.pixel_array
    except (AttributeError, ValueError, TypeError) as exc:
        raise ImageReadError(f"El DICOM '{ruta}' no contiene un PixelData válido: {exc}") from exc

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixel_array = pixel_array.astype(np.float64) * slope + intercept

    photometric = getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")
    gray_uint8 = _normalize_to_uint8(pixel_array)

    if photometric == "MONOCHROME1":
        logger.info(
            "PhotometricInterpretation='MONOCHROME1' detectado en '%s'; invirtiendo intensidades.",
            ruta,
        )
        gray_uint8 = 255 - gray_uint8
    elif photometric != "MONOCHROME2":
        logger.warning(
            "PhotometricInterpretation '%s' en '%s' no es MONOCHROME1/2; "
            "se procesa como escala de grises sin inversión.",
            photometric,
            ruta,
        )

    rgb_array = _gray_to_rgb(gray_uint8)
    imagen_pil = Image.fromarray(rgb_array, mode="RGB")
    return rgb_array, imagen_pil


def read_jpg_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Lee un archivo de imagen estándar (JPG, PNG, etc.) usando PIL.

    Convierte la imagen a modo "RGB" independientemente de su modo original
    (escala de grises, RGBA, paleta, etc.) para cumplir el contrato de datos
    del proyecto.

    Args:
        path: Ruta al archivo de imagen.

    Returns:
        tuple[np.ndarray, PIL.Image.Image]: Una tupla con el arreglo RGB
        ``uint8`` de forma (H, W, 3) y la imagen PIL equivalente en modo
        "RGB".

    Raises:
        FileNotFoundError: Si ``path`` no existe en el sistema de archivos.
        ImageReadError: Si el archivo existe pero PIL no puede decodificarlo.

    Example:
        >>> arreglo, imagen = read_jpg_file("radiografia.jpg")
        >>> arreglo.dtype
        dtype('uint8')
    """
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo de imagen no existe: {ruta}")

    logger.info("Leyendo archivo de imagen: %s", ruta)
    try:
        with Image.open(ruta) as imagen_original:
            imagen_rgb = imagen_original.convert("RGB")
            imagen_rgb.load()
    except (OSError, ValueError) as exc:
        raise ImageReadError(f"No se pudo leer la imagen '{ruta}': {exc}") from exc

    array_rgb = np.asarray(imagen_rgb, dtype=np.uint8)
    return array_rgb, imagen_rgb


def read_image_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Despacha la lectura de un archivo según su extensión.

    Consulta ``config.SUPPORTED_EXTENSIONS`` para validar que la extensión
    (en minúsculas) esté soportada. Las extensiones DICOM (``.dcm``,
    ``.dicom``) se enrutan a ``read_dicom_file``; el resto de extensiones
    soportadas se enrutan a ``read_jpg_file`` (basado en PIL).

    Args:
        path: Ruta al archivo de imagen o DICOM.

    Returns:
        tuple[np.ndarray, PIL.Image.Image]: Una tupla con el arreglo RGB
        ``uint8`` de forma (H, W, 3) y la imagen PIL equivalente.

    Raises:
        FileNotFoundError: Si ``path`` no existe en el sistema de archivos.
        UnsupportedFormatError: Si la extensión no está en
            ``config.SUPPORTED_EXTENSIONS``.
        ImageReadError: Si el archivo tiene una extensión soportada pero está
            corrupto o no puede decodificarse.

    Example:
        >>> arreglo, imagen = read_image_file("estudio.dcm")
        >>> arreglo.shape
        (512, 512, 3)
    """
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo no existe: {ruta}")

    sufijo = ruta.suffix.lower()
    if sufijo not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Extensión no soportada '{sufijo}'. Soportadas: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if sufijo in DICOM_EXTENSIONS:
        return read_dicom_file(ruta)
    return read_jpg_file(ruta)
