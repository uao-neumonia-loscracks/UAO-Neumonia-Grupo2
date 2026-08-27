"""Módulo integrator — CAPA CONTROLADOR (orquestador).

Único punto de contacto entre la Vista (``detector_neumonia.py``) y el resto
del sistema (Modelo + Controlador de bajo nivel). Orquesta la predicción de
neumonía a partir de una imagen ya leída, delegando el preprocesamiento y la
inferencia a ``grad_cam.grad_cam``.

Advertencia clínica:
    Esta herramienta es de apoyo educativo. NO es un dispositivo médico
    certificado y no debe usarse para tomar decisiones clínicas reales.
"""

import logging
from dataclasses import dataclass

import numpy as np

from config import CLASS_LABELS, IMAGE_SIZE
from grad_cam import grad_cam

logger = logging.getLogger(__name__)

_NDIM_IMAGEN_ESPERADO = 3
_NUM_CANALES_RGB = 3
_DTYPE_IMAGEN_ENTRADA = np.uint8
_DTYPE_HEATMAP = np.uint8
_PROBABILIDAD_MINIMA = 0.0
_PROBABILIDAD_MAXIMA = 100.0
_FORMA_HEATMAP_ESPERADA: tuple[int, int, int] = (*IMAGE_SIZE, _NUM_CANALES_RGB)


class PredictionError(RuntimeError):
    """Envuelve fallos internos de la orquestación de predicción con contexto.

    Permite que la Vista capture un único tipo de excepción y muestre un
    mensaje amigable al usuario final, sin conocer los detalles internos de
    TensorFlow, Grad-CAM o el mapeo de clases.

    Attributes:
        causa: Excepción original que originó este error, o ``None`` si no
            aplica (por ejemplo, cuando el propio ``integrator`` detecta el
            problema sin una excepción subyacente).
    """

    def __init__(self, mensaje: str, causa: Exception | None = None) -> None:
        """Inicializa la excepción con un mensaje legible y su causa opcional.

        Args:
            mensaje: Descripción del fallo, orientada al usuario final.
            causa: Excepción original capturada, para trazabilidad en logs.
        """
        super().__init__(mensaje)
        self.causa = causa


@dataclass(frozen=True)
class PredictionResult:
    """Resultado inmutable de una predicción de neumonía.

    Attributes:
        label: Etiqueta legible de la clase predicha (``"bacteriana"``,
            ``"normal"`` o ``"viral"``, según ``config.CLASS_LABELS``).
        probability: Probabilidad de la clase predicha, en porcentaje
            (rango ``[0.0, 100.0]``).
        class_index: Índice entero de la clase predicha (``0``, ``1`` o
            ``2``), tal como lo define ``config.CLASS_LABELS``.
        heatmap: Overlay Grad-CAM en RGB ``uint8`` con forma
            ``(512, 512, 3)``, listo para mostrarse en la Vista.
    """

    label: str
    probability: float
    class_index: int
    heatmap: np.ndarray

    def __post_init__(self) -> None:
        """Valida los invariantes del resultado tras la construcción.

        Raises:
            ValueError: Si `probability` está fuera de `[0, 100]`, si
                `class_index` no es una clave válida de
                `config.CLASS_LABELS`, o si `heatmap` no tiene la forma o
                el tipo de dato esperados.
        """
        if not (_PROBABILIDAD_MINIMA <= self.probability <= _PROBABILIDAD_MAXIMA):
            raise ValueError(
                f"probability={self.probability!r} fuera de rango "
                f"[{_PROBABILIDAD_MINIMA}, {_PROBABILIDAD_MAXIMA}]."
            )
        if self.class_index not in CLASS_LABELS:
            raise ValueError(
                f"class_index={self.class_index!r} no está en CLASS_LABELS "
                f"({sorted(CLASS_LABELS)})."
            )
        if not isinstance(self.heatmap, np.ndarray):
            raise ValueError(f"heatmap debe ser np.ndarray, se recibió {type(self.heatmap)!r}.")
        if self.heatmap.shape != _FORMA_HEATMAP_ESPERADA:
            raise ValueError(
                f"heatmap.shape={self.heatmap.shape!r} distinto de {_FORMA_HEATMAP_ESPERADA!r}."
            )
        if self.heatmap.dtype != _DTYPE_HEATMAP:
            raise ValueError(
                f"heatmap.dtype={self.heatmap.dtype!r} distinto de {_DTYPE_HEATMAP!r}."
            )


def _validar_imagen_entrada(image_rgb: np.ndarray) -> None:
    """Valida que la imagen de entrada cumpla el contrato RGB uint8.

    Args:
        image_rgb: Arreglo candidato a imagen de radiografía en RGB.

    Raises:
        ValueError: Si `image_rgb` no es `np.ndarray`, no tiene 3 dimensiones,
            no tiene 3 canales, o su `dtype` no es `uint8`.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError(f"image_rgb debe ser np.ndarray, se recibió {type(image_rgb)!r}.")
    if image_rgb.ndim != _NDIM_IMAGEN_ESPERADO:
        raise ValueError(
            f"image_rgb debe tener {_NDIM_IMAGEN_ESPERADO} dimensiones (H, W, C), "
            f"tiene ndim={image_rgb.ndim}."
        )
    if image_rgb.shape[2] != _NUM_CANALES_RGB:
        raise ValueError(
            f"image_rgb debe tener {_NUM_CANALES_RGB} canales, tiene shape[2]={image_rgb.shape[2]}."
        )
    if image_rgb.dtype != _DTYPE_IMAGEN_ENTRADA:
        raise ValueError(
            f"image_rgb debe ser dtype={_DTYPE_IMAGEN_ENTRADA!r}, es dtype={image_rgb.dtype!r}."
        )


def predict(image_rgb: np.ndarray) -> PredictionResult:
    """Orquesta la predicción de neumonía sobre una imagen ya leída.

    Flujo interno:
        1. Valida la entrada (``ndarray`` RGB `uint8` de 3 canales). Si no
           cumple el contrato, lanza `ValueError` descriptivo de inmediato
           (falla rápido, antes de tocar el Modelo).
        2. NO se llama a `preprocess_img.preprocess()` en este módulo.
           Decisión D6.1: `grad_cam.grad_cam(image_rgb)` ya ejecuta
           internamente el preprocesamiento (contrato 5.5, heredado de la
           decisión D5.2 del Hilo H5) y el forward pass del modelo. Volver a
           preprocesar aquí duplicaría lógica sin necesidad y violaría la
           instrucción explícita heredada en el handoff H5 ("no repetir
           lógica de preprocesamiento ni de carga del modelo, grad_cam() ya
           lo hace"). Esta decisión fue confirmada explícitamente por el
           usuario en este hilo.
        3. Llama a `grad_cam.grad_cam(image_rgb)`, que devuelve
           `(overlay, class_index, probability)`. Cualquier fallo interno
           (TensorFlow, forma de tensores, etc.) se envuelve en
           `PredictionError` para no filtrar detalles internos a la Vista.
        4. Mapea `class_index` a una etiqueta legible usando
           `config.CLASS_LABELS`. Si el índice no existe en el mapeo, se
           lanza `PredictionError` en vez de un `KeyError` críptico.
        5. Construye y devuelve un `PredictionResult` inmutable. Cualquier
           violación de sus invariantes (`__post_init__`) también se
           envuelve en `PredictionError`.

    Args:
        image_rgb: Imagen de radiografía en RGB, `np.ndarray` `uint8` con
            forma `(H, W, 3)`, tal como la devuelve `read_img.read_image_file`.

    Returns:
        `PredictionResult` inmutable con la etiqueta, probabilidad, índice de
        clase y overlay Grad-CAM.

    Raises:
        ValueError: Si `image_rgb` no cumple el contrato de entrada.
        PredictionError: Si falla la inferencia/Grad-CAM, si `class_index`
            no está en `config.CLASS_LABELS`, o si el resultado construido
            no cumple sus invariantes.
    """
    _validar_imagen_entrada(image_rgb)

    try:
        overlay, class_index, probability = grad_cam(image_rgb)
    except Exception as exc:  # noqa: BLE001 (frontera de la capa; se re-envuelve)
        raise PredictionError("No fue posible generar la predicción Grad-CAM.", causa=exc) from exc

    try:
        label = CLASS_LABELS[class_index]
    except KeyError as exc:
        raise PredictionError(
            f"Índice de clase desconocido devuelto por el modelo: {class_index!r}. "
            f"Índices válidos: {sorted(CLASS_LABELS)}.",
            causa=exc,
        ) from exc

    try:
        resultado = PredictionResult(
            label=label,
            probability=float(probability),
            class_index=int(class_index),
            heatmap=overlay,
        )
    except ValueError as exc:
        raise PredictionError(
            f"El resultado de la predicción no cumple los invariantes esperados: {exc}",
            causa=exc,
        ) from exc

    logger.info("Predicción completada: label=%s probability=%.2f%%", label, probability)
    return resultado
