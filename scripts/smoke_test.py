"""CLI de inferencia end-to-end sin interfaz gráfica.

Sirve para verificar el contenedor Docker sin necesidad de un servidor X11.
Admite un modo `--dry-run` que valida lectura y preprocesamiento sin requerir
el modelo real (útil en CI, donde `conv_MLP_84.h5` no está disponible).

Advertencia: esta herramienta es de apoyo educativo, NO es un dispositivo
médico certificado. No debe usarse para decisiones clínicas reales.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from integrator import PredictionError, PredictionResult, predict  # noqa: E402
from preprocess_img import preprocess  # noqa: E402
from read_img import ImageReadError, UnsupportedFormatError, read_image_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los atributos `image`, `save_out` y `dry_run`.
    """
    parser = argparse.ArgumentParser(
        description="Smoke test de inferencia end-to-end sin interfaz gráfica."
    )
    parser.add_argument(
        "--image", required=True, type=Path, help="Ruta a la imagen (DICOM/JPG/PNG)."
    )
    parser.add_argument(
        "--save-out",
        type=Path,
        default=None,
        help="Ruta donde guardar el overlay de Grad-CAM (PNG). Opcional.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Omite la inferencia con el modelo real; solo valida lectura y preprocesamiento.",
    )
    return parser.parse_args()


def run_smoke_test(
    image_path: Path, save_out: Path | None, dry_run: bool
) -> PredictionResult | None:
    """Ejecuta el pipeline de inferencia sobre una imagen.

    Args:
        image_path: ruta al archivo de imagen soportado.
        save_out: ruta opcional donde guardar el overlay de Grad-CAM.
        dry_run: si es True, omite la llamada a `predict` y solo valida
            lectura y preprocesamiento (no requiere el modelo real).

    Returns:
        El resultado de la predicción, o None si `dry_run` es True.

    Raises:
        UnsupportedFormatError: si la extensión no está soportada.
        ImageReadError: si el archivo no se pudo leer.
        PredictionError: si el pipeline de predicción falla.
    """
    logger.info("Leyendo imagen: %s", image_path)
    image_rgb, _ = read_image_file(image_path)

    if dry_run:
        logger.info("Modo --dry-run: se omite la inferencia con el modelo real.")
        batch = preprocess(image_rgb)
        logger.info("Preprocesamiento OK -> shape=%s dtype=%s", batch.shape, batch.dtype)
        return None

    logger.info("Ejecutando predicción...")
    result = predict(image_rgb)
    logger.info(
        "Clase=%s (index=%d) probabilidad=%.2f%%",
        result.label,
        result.class_index,
        result.probability,
    )

    if save_out is not None:
        save_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.heatmap).save(save_out)
        logger.info("Overlay guardado en: %s", save_out)

    return result


def main() -> None:
    """Punto de entrada del script."""
    args = parse_args()
    try:
        run_smoke_test(args.image, args.save_out, args.dry_run)
    except (UnsupportedFormatError, ImageReadError, PredictionError, ValueError) as exc:
        logger.exception("Fallo el smoke test: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
