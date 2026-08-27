"""Módulo del MODELO: carga, caché y verificación de integridad de la CNN.

Capa MVC: MODELO. Este módulo es el ÚNICO punto (junto con grad_cam.py) donde
vive TensorFlow/Keras en todo el proyecto. No debe ser importado nunca desde
la Vista (src/detector_neumonia.py).

Veredicto de compatibilidad (Hilo H0, decisión congelada en el plan maestro,
sección 1.2): **Plan B**. `tensorflow==2.20.*` trae Keras 3 por defecto, que
NO carga de forma confiable modelos `.h5` legacy (Keras 2, formato HDF5
clásico como `conv_MLP_84.h5`). Por eso se fuerza la implementación legacy
de Keras mediante la variable de entorno `TF_USE_LEGACY_KERAS=1` (requiere
el paquete `tf-keras` instalado); con esa variable activa, `tensorflow.keras`
resuelve internamente a `tf_keras`, que sí sabe leer el formato HDF5 legacy.
NO se usa `import keras` (eso apuntaría a Keras 3 puro) ni se improvisa
`spike_compat.py` (Plan A) — ese script quedó descartado por el veredicto.

Aviso clínico: esta herramienta es de apoyo EDUCATIVO. NO es un dispositivo
médico certificado y no debe usarse para tomar decisiones clínicas reales.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

# Silencia el ruido de logs C++ de TensorFlow (backend nativo). Debe fijarse
# ANTES del primer `import tensorflow`. Niveles: "0"=todo, "1"=oculta INFO,
# "2"=oculta INFO+WARNING, "3"=oculta INFO+WARNING+ERROR. Usamos "2" para no
# perder errores reales de C++ pero sí silenciar el spam de inicialización
# (cuDNN/cuBLAS/oneDNN) que no aporta nada en un entorno sin GPU dedicada.
# `setdefault` respeta si el usuario ya fijó la variable en su shell.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Fuerza la implementación legacy de Keras (Plan B del handoff H0). Debe fijarse
# ANTES del primer `import tensorflow` para que tensorflow.keras resuelva a
# tf_keras internamente, en vez de a Keras 3 puro.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# GOTCHA descubierto empíricamente en H4: al importar `tensorflow` con
# TF_USE_LEGACY_KERAS=1, TensorFlow importa `tf_keras` como parte de su propio
# `__init__.py`. El módulo `tf_keras/src/losses.py` referencia, A NIVEL DE
# MÓDULO (se ejecuta durante el import, no al llamar a nuestras funciones),
# el símbolo legacy `tf.losses.sparse_softmax_cross_entropy`, lo cual dispara
# un `WARNING:tensorflow:` de deprecación INMEDIATAMENTE durante la línea
# `import tensorflow as tf` de abajo. Por eso `tf.get_logger().setLevel(...)`
# NO sirve si se llama DESPUÉS del import: el aviso ya se emitió. La solución
# es configurar, con el módulo estándar `logging` (que no requiere que
# TensorFlow ya esté importado), el nivel del logger con nombre "tensorflow"
# ANTES del import, para que cuando TensorFlow llame internamente a
# `logging.getLogger("tensorflow").warning(...)` durante su propia
# inicialización, ese logger ya esté configurado en ERROR y descarte el
# aviso. Esto es localizado (un logger con nombre específico) y documentado;
# NO es un `warnings.filterwarnings("ignore")` global.
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf  # noqa: E402 (import diferido: requiere las env vars y el logger previos)

import config  # noqa: E402

if TYPE_CHECKING:
    from tf_keras import Model as KerasModel
else:
    KerasModel = tf.keras.Model

# Alias público de Keras (legacy, vía TF_USE_LEGACY_KERAS=1) para que otros
# módulos de la capa Modelo (p. ej. tests, grad_cam.py) puedan referenciarlo
# sin repetir el manejo de variables de entorno.
keras = tf.keras

logger = logging.getLogger(__name__)

# Tipos de capa convolucional que se consideran candidatas para Grad-CAM.
_CAPAS_CONVOLUCIONALES: tuple[type, ...] = (
    keras.layers.Conv2D,
    keras.layers.SeparableConv2D,
)


class ModelNotFoundError(FileNotFoundError):
    """Se lanza cuando el archivo del modelo `.h5` no existe en la ruta resuelta.

    Attributes:
        model_path: Ruta absoluta en la que se buscó el archivo del modelo.
    """

    def __init__(self, model_path: Path) -> None:
        """Construye el error con un mensaje accionable para el usuario.

        Args:
            model_path: Ruta absoluta en la que se buscó el archivo `.h5`.
        """
        self.model_path = model_path
        mensaje = (
            f"No se encontró el archivo del modelo en: {model_path}\n"
            "Cómo resolverlo:\n"
            "  1. Define la variable de entorno MODEL_PATH apuntando al archivo "
            "conv_MLP_84.h5, por ejemplo:\n"
            '     export MODEL_PATH="/ruta/completa/a/conv_MLP_84.h5"   (Linux/macOS)\n'
            '     $env:MODEL_PATH="C:\\ruta\\completa\\a\\conv_MLP_84.h5"  (PowerShell)\n'
            "  2. O coloca el archivo en la carpeta por defecto: "
            f"{config.get_project_root() / 'models' / config.DEFAULT_MODEL_FILENAME}\n"
            "  3. Consulta models/README.md para instrucciones de descarga y el "
            "SHA256 esperado del modelo."
        )
        super().__init__(mensaje)


def compute_sha256(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo, leyéndolo por bloques.

    Args:
        path: Ruta al archivo binario a verificar.

    Returns:
        El hash SHA-256 en hexadecimal, en minúsculas.

    Raises:
        FileNotFoundError: Si `path` no existe.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se puede calcular SHA-256: no existe {path}")

    hasher = hashlib.sha256()
    tamano_bloque = 65536
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(tamano_bloque), b""):
            hasher.update(bloque)
    return hasher.hexdigest()


def verify_model_integrity(path: Path, expected_sha256: str | None = None) -> bool:
    """Verifica la integridad del archivo del modelo mediante su hash SHA-256.

    Si `expected_sha256` es `None`, solo se comprueba que el archivo exista y
    que su hash pueda calcularse sin error (integridad de lectura), y se
    registra el hash calculado en el log para que quede trazado. Si se provee
    `expected_sha256`, se compara (sin distinguir mayúsculas/minúsculas) contra
    el hash real.

    Args:
        path: Ruta al archivo `.h5` del modelo.
        expected_sha256: Hash SHA-256 esperado, en hexadecimal. Si es `None`,
            solo se valida que el archivo sea legible.

    Returns:
        `True` si el archivo existe, es legible y (cuando se provee) su hash
        coincide con `expected_sha256`. `False` en caso contrario.
    """
    path = Path(path)
    try:
        hash_real = compute_sha256(path)
    except FileNotFoundError:
        logger.warning("Verificación de integridad fallida: no existe %s", path)
        return False

    if expected_sha256 is None:
        logger.info("SHA-256 de %s: %s (sin hash de referencia para comparar)", path, hash_real)
        return True

    coincide = hash_real.lower() == expected_sha256.lower()
    if not coincide:
        logger.warning(
            "SHA-256 de %s no coincide. Esperado=%s Real=%s", path, expected_sha256, hash_real
        )
    return coincide


@lru_cache(maxsize=1)
def load_cnn_model(model_path: str | None = None) -> KerasModel:
    """Carga la CNN entrenada desde el archivo `.h5` y cachea el resultado.

    Se decora con `functools.lru_cache(maxsize=1)` porque cargar el archivo
    `.h5` (arquitectura + pesos) toma varios segundos, y la GUI de Tkinter
    puede invocar esta función una vez por cada clic en "Predecir": sin caché,
    cada predicción re-leería el binario completo desde disco. `maxsize=1`
    basta porque en producción solo se usa una ruta de modelo por proceso; si
    se necesita forzar una recarga (p. ej. tras actualizar el `.h5`, o en
    tests que cambian `MODEL_PATH`), se debe llamar explícitamente a
    `load_cnn_model.cache_clear()` antes de la siguiente invocación.

    Se carga con `compile=False` porque en inferencia no se necesita el
    optimizador ni las métricas de entrenamiento asociadas al modelo
    original; intentar recompilar con métricas legacy (definidas con una
    versión antigua de Keras) puede disparar warnings de deserialización
    que no aportan nada en un flujo de solo-inferencia como el de esta app.

    Args:
        model_path: Ruta al archivo `.h5`. Si es `None`, se resuelve con
            `config.get_model_path()` (variable de entorno `MODEL_PATH` con
            fallback a `models/conv_MLP_84.h5`).

    Returns:
        La instancia de `keras.Model` cargada y lista para inferencia.

    Raises:
        ModelNotFoundError: Si el archivo resuelto no existe en disco.
    """
    ruta_resuelta = Path(model_path) if model_path is not None else config.get_model_path()

    if not ruta_resuelta.is_file():
        raise ModelNotFoundError(ruta_resuelta)

    logger.info("Cargando modelo desde %s ...", ruta_resuelta)
    inicio = time.perf_counter()
    modelo = keras.models.load_model(ruta_resuelta, compile=False)
    duracion = time.perf_counter() - inicio
    logger.info("Modelo cargado en %.3f s (%s capas)", duracion, len(modelo.layers))

    return modelo


def _buscar_ultima_conv_recursiva(layers: list) -> str | None:
    """Recorre una lista de capas en reversa buscando la última convolucional.

    Si una capa es a su vez un submodelo (contiene su propio atributo
    `layers`), se explora recursivamente ese submodelo también en reversa
    antes de continuar con las capas restantes del nivel actual.

    Args:
        layers: Lista de capas de Keras (`model.layers` o `submodel.layers`).

    Returns:
        El nombre de la primera capa convolucional encontrada recorriendo en
        reversa, o `None` si no se encontró ninguna en este nivel ni en sus
        submodelos anidados.
    """
    for capa in reversed(layers):
        sublayers = getattr(capa, "layers", None)
        if sublayers:
            encontrada = _buscar_ultima_conv_recursiva(list(sublayers))
            if encontrada is not None:
                return encontrada
        if isinstance(capa, _CAPAS_CONVOLUCIONALES):
            return capa.name
    return None


def get_last_conv_layer_name(model: KerasModel) -> str:
    """Devuelve el nombre de la última capa convolucional del modelo.

    Recorre `model.layers` en orden inverso (de la salida hacia la entrada)
    buscando una instancia de `Conv2D` o `SeparableConv2D`. Si el modelo tiene
    submodelos anidados (por ejemplo, bloques encapsulados como `keras.Model`
    o `keras.Sequential` dentro del modelo principal), se explora también
    dentro de ellos de forma recursiva. Esta capa es el objetivo estándar
    para Grad-CAM, ya que conserva la información espacial más rica antes de
    las capas de pooling/clasificación.

    Args:
        model: Modelo Keras ya cargado.

    Returns:
        El nombre (`str`) de la última capa convolucional encontrada.

    Raises:
        ValueError: Si no se encuentra ninguna capa `Conv2D`/`SeparableConv2D`
            en todo el modelo (ni en submodelos anidados).
    """
    nombre = _buscar_ultima_conv_recursiva(list(model.layers))
    if nombre is None:
        raise ValueError(
            "No se encontró ninguna capa Conv2D/SeparableConv2D en el modelo "
            f"'{model.name}'. Grad-CAM requiere al menos una capa convolucional."
        )
    return nombre


def model_summary_text(model: KerasModel) -> str:
    """Captura `model.summary()` como texto, para el Model Card y la sustentación.

    Args:
        model: Modelo Keras ya cargado.

    Returns:
        El resumen de arquitectura del modelo como una única cadena de texto,
        con saltos de línea entre capas (igual a lo que `summary()` imprime
        por consola, pero capturado en lugar de impreso).
    """
    lineas: list[str] = []
    model.summary(print_fn=lineas.append)
    return "\n".join(lineas)
