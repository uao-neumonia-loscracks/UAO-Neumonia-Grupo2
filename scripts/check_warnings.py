"""Verifica que el código propio de src/ no emita warnings.

Importa todos los módulos de `src/` y, si `MODEL_PATH` existe, ejecuta una
inferencia end-to-end con datos sintéticos bajo captura de warnings. Falla
(exit 1) si algún warning se origina en un archivo dentro de `src/`. Los
warnings de librerías de terceros (TensorFlow, protobuf, etc.) se listan
pero no hacen fallar el script.

Advertencia: esta herramienta es de apoyo educativo, NO es un dispositivo
médico certificado.
"""

from __future__ import annotations

import importlib
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

MODULES = ["config", "read_img", "preprocess_img", "load_model", "grad_cam", "integrator"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _is_from_src(warning: warnings.WarningMessage) -> bool:
    """Determina si un warning se originó en un archivo dentro de src/.

    Args:
        warning: el registro de warning capturado.

    Returns:
        True si el archivo de origen del warning está dentro de src/.
    """
    try:
        origin = Path(str(warning.filename)).resolve()
    except (OSError, ValueError):
        return False
    return origin == SRC_DIR or SRC_DIR in origin.parents


def check_imports() -> None:
    """Importa todos los módulos de src/ (dispara warnings de import, si hay)."""
    for name in MODULES:
        logger.info("Importando módulo: %s", name)
        importlib.import_module(name)


def run_inference_if_possible() -> bool:
    """Ejecuta una inferencia sintética si el modelo está disponible.

    Recupera `config` e `integrator` desde `sys.modules` vía
    `importlib.import_module` (ya fueron importados por `check_imports`) en
    lugar de una sentencia `import` anidada, para no violar la regla de
    estilo que exige imports a nivel de módulo (Ruff PLC0415).

    Returns:
        True si la inferencia se ejecutó, False si se omitió por falta de modelo.
    """
    config = importlib.import_module("config")
    integrator = importlib.import_module("integrator")

    model_path = config.get_model_path()
    if not model_path.exists():
        logger.warning(
            "MODEL_PATH no existe (%s): se omite la verificación de warnings "
            "en tiempo de inferencia. Solo se valida la fase de importación.",
            model_path,
        )
        return False

    synthetic_rgb = np.random.default_rng(seed=42).integers(
        0, 256, size=(512, 512, 3), dtype=np.uint8
    )
    integrator.predict(synthetic_rgb)
    return True


def main() -> None:
    """Punto de entrada del script."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_imports()
        ran_inference = run_inference_if_possible()

    own_warnings = [w for w in caught if _is_from_src(w)]
    third_party_warnings = [w for w in caught if not _is_from_src(w)]

    if third_party_warnings:
        logger.info("Warnings de terceros detectados (no bloquean el build):")
        for w in third_party_warnings:
            logger.info("  %s:%d: %s: %s", w.filename, w.lineno, w.category.__name__, w.message)

    if own_warnings:
        logger.error("Warnings originados en src/ (BLOQUEANTE):")
        for w in own_warnings:
            logger.error("  %s:%d: %s: %s", w.filename, w.lineno, w.category.__name__, w.message)
        sys.exit(1)

    suffix = " (con inferencia real)" if ran_inference else " (solo importación, sin modelo)"
    logger.info("0 warnings originados en src/%s", suffix)
    sys.exit(0)


if __name__ == "__main__":
    main()
