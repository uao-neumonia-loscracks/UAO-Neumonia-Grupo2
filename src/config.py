"""Configuración centralizada del proyecto UAO-Neumonia.

Capa MVC: transversal (utilizado por Modelo, Controlador y Vista).
No importa tensorflow, keras ni tkinter: solo constantes y resolución de rutas.
"""

import os
from pathlib import Path

IMAGE_SIZE: tuple[int, int] = (512, 512)
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (4, 4)

# Mapeo de clases declarado por el autor original. Debe verificarse empíricamente
# con imágenes reales de clase conocida (TODO heredado de H0, pendiente de cierre).
CLASS_LABELS: dict[int, str] = {0: "bacteriana", 1: "normal", 2: "viral"}

DEFAULT_MODEL_FILENAME: str = "conv_MLP_84.h5"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".dcm", ".dicom", ".jpg", ".jpeg", ".png")

GRAD_CAM_TARGET_LAYER: str = "conv10_thisone"

MODEL_PATH_ENV_VAR: str = "MODEL_PATH"


def get_project_root() -> Path:
    """Devuelve la ruta absoluta a la raíz del repositorio.

    La raíz se calcula como el directorio padre de la carpeta ``src`` que
    contiene este archivo, asumiendo el layout plano definido en el plan
    maestro (``package-dir = {"" = "src"}``).

    Returns:
        Path: Ruta absoluta al directorio raíz del proyecto.
    """
    return Path(__file__).resolve().parent.parent


def get_model_path() -> Path:
    """Resuelve la ruta absoluta del archivo del modelo entrenado.

    Prioriza la variable de entorno ``MODEL_PATH``. Si no está definida,
    recurre al valor por defecto ``<raiz_proyecto>/models/conv_MLP_84.h5``.
    Esta función NO valida que el archivo exista; esa responsabilidad
    corresponde a ``load_model.load_cnn_model``.

    Returns:
        Path: Ruta absoluta resuelta hacia el archivo ``.h5`` del modelo.
    """
    env_value = os.environ.get(MODEL_PATH_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return get_project_root() / "models" / DEFAULT_MODEL_FILENAME
