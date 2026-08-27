"""Pruebas unitarias para el módulo controlador ``read_img``.

Cubre lectura de DICOM (MONOCHROME1/2), lectura de imágenes estándar (JPG,
PNG), el despachador ``read_image_file`` y el manejo de errores esperados
(archivo inexistente, extensión no soportada, archivo corrupto).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import read_img
from config import SUPPORTED_EXTENSIONS

CANALES_RGB = 3
VALOR_MAXIMO_UINT8 = 255
VALOR_MINIMO_UINT8 = 0
DIMENSION_DICOM_SINTETICO = 512
DIMENSION_IMAGEN_SINTETICA = 64


def _assertar_contrato_rgb(arreglo: np.ndarray, imagen: Image.Image) -> None:
    """Verifica que un resultado cumpla el contrato de datos de ``read_img``.

    Args:
        arreglo: Arreglo devuelto por una función de lectura.
        imagen: Imagen PIL devuelta por una función de lectura.
    """
    assert arreglo.ndim == CANALES_RGB
    assert arreglo.shape[2] == CANALES_RGB
    assert arreglo.dtype == np.uint8
    assert arreglo.min() >= VALOR_MINIMO_UINT8
    assert arreglo.max() <= VALOR_MAXIMO_UINT8
    assert isinstance(imagen, Image.Image)
    assert imagen.mode == "RGB"


class TestReadDicomFile:
    """Pruebas para ``read_dicom_file``."""

    def test_lee_dicom_monochrome2_shape_dtype_rango(self, synthetic_dicom: Path) -> None:
        """El DICOM MONOCHROME2 sintético produce un RGB (512, 512, 3) uint8."""
        arreglo, imagen = read_img.read_dicom_file(synthetic_dicom)
        _assertar_contrato_rgb(arreglo, imagen)
        assert arreglo.shape == (
            DIMENSION_DICOM_SINTETICO,
            DIMENSION_DICOM_SINTETICO,
            CANALES_RGB,
        )

    def test_canales_rgb_identicos(self, synthetic_dicom: Path) -> None:
        """Los tres canales RGB deben ser idénticos (replicación de escala de grises)."""
        arreglo, _ = read_img.read_dicom_file(synthetic_dicom)
        assert np.array_equal(arreglo[:, :, 0], arreglo[:, :, 1])
        assert np.array_equal(arreglo[:, :, 1], arreglo[:, :, 2])

    def test_inversion_monochrome1(
        self, synthetic_dicom: Path, synthetic_dicom_monochrome1: Path
    ) -> None:
        """MONOCHROME1 debe producir el complemento (255 - x) de MONOCHROME2."""
        arreglo_mono2, _ = read_img.read_dicom_file(synthetic_dicom)
        arreglo_mono1, _ = read_img.read_dicom_file(synthetic_dicom_monochrome1)
        gris_mono2 = arreglo_mono2[:, :, 0].astype(np.int16)
        gris_mono1 = arreglo_mono1[:, :, 0].astype(np.int16)
        assert np.array_equal(gris_mono1, VALOR_MAXIMO_UINT8 - gris_mono2)

    def test_normalizacion_usa_rango_real(self, synthetic_dicom: Path) -> None:
        """El patrón de gradiente sintético debe cubrir aproximadamente todo el rango."""
        arreglo, _ = read_img.read_dicom_file(synthetic_dicom)
        assert arreglo.min() == VALOR_MINIMO_UINT8
        assert arreglo.max() == VALOR_MAXIMO_UINT8

    def test_archivo_inexistente_lanza_filenotfound(self, tmp_path: Path) -> None:
        """Un DICOM inexistente debe lanzar ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            read_img.read_dicom_file(tmp_path / "no_existe.dcm")

    def test_dicom_corrupto_lanza_imagereaderror(self, corrupt_file: Path) -> None:
        """Un archivo .dcm corrupto debe lanzar ``ImageReadError`` encadenado."""
        with pytest.raises(read_img.ImageReadError) as info:
            read_img.read_dicom_file(corrupt_file)
        assert info.value.__cause__ is not None


class TestReadJpgFile:
    """Pruebas para ``read_jpg_file``."""

    def test_lee_jpg_shape_dtype_rango(self, synthetic_jpg: Path) -> None:
        """El JPG sintético produce un arreglo RGB uint8 válido."""
        arreglo, imagen = read_img.read_jpg_file(synthetic_jpg)
        _assertar_contrato_rgb(arreglo, imagen)
        assert arreglo.shape == (
            DIMENSION_IMAGEN_SINTETICA,
            DIMENSION_IMAGEN_SINTETICA,
            CANALES_RGB,
        )

    def test_lee_png_convertido_a_rgb(self, synthetic_png: Path) -> None:
        """Un PNG en escala de grises debe convertirse a RGB de 3 canales."""
        arreglo, imagen = read_img.read_jpg_file(synthetic_png)
        _assertar_contrato_rgb(arreglo, imagen)
        assert arreglo.shape == (
            DIMENSION_IMAGEN_SINTETICA,
            DIMENSION_IMAGEN_SINTETICA,
            CANALES_RGB,
        )

    def test_archivo_inexistente_lanza_filenotfound(self, tmp_path: Path) -> None:
        """Una imagen inexistente debe lanzar ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            read_img.read_jpg_file(tmp_path / "no_existe.jpg")

    def test_imagen_corrupta_lanza_imagereaderror(self, tmp_path: Path) -> None:
        """Un archivo .jpg con contenido inválido debe lanzar ``ImageReadError``."""
        ruta = tmp_path / "corrupta.jpg"
        ruta.write_bytes(b"no-es-una-imagen-valida" * 5)
        with pytest.raises(read_img.ImageReadError) as info:
            read_img.read_jpg_file(ruta)
        assert info.value.__cause__ is not None


class TestReadImageFile:
    """Pruebas para el despachador ``read_image_file``."""

    def test_despacha_dicom_correctamente(self, synthetic_dicom: Path) -> None:
        """Un archivo .dcm debe despacharse a la ruta DICOM."""
        arreglo, imagen = read_img.read_image_file(synthetic_dicom)
        _assertar_contrato_rgb(arreglo, imagen)
        assert arreglo.shape == (
            DIMENSION_DICOM_SINTETICO,
            DIMENSION_DICOM_SINTETICO,
            CANALES_RGB,
        )

    def test_despacha_jpg_correctamente(self, synthetic_jpg: Path) -> None:
        """Un archivo .jpg debe despacharse a la ruta de imagen estándar."""
        arreglo, imagen = read_img.read_image_file(synthetic_jpg)
        _assertar_contrato_rgb(arreglo, imagen)

    def test_archivo_inexistente_lanza_filenotfound(self, tmp_path: Path) -> None:
        """Un archivo inexistente debe lanzar ``FileNotFoundError`` sin importar extensión."""
        with pytest.raises(FileNotFoundError):
            read_img.read_image_file(tmp_path / "fantasma.dcm")

    @pytest.mark.parametrize("extension", [".txt", ".pdf", ".exe", ".docx", ".gif"])
    def test_extension_no_soportada_lanza_unsupportedformaterror(
        self, tmp_path: Path, extension: str
    ) -> None:
        """Extensiones fuera de ``config.SUPPORTED_EXTENSIONS`` deben ser rechazadas."""
        assert extension not in SUPPORTED_EXTENSIONS
        ruta = tmp_path / f"archivo{extension}"
        ruta.write_bytes(b"contenido-arbitrario")
        with pytest.raises(read_img.UnsupportedFormatError):
            read_img.read_image_file(ruta)

    def test_dicom_corrupto_via_dispatcher_lanza_imagereaderror(self, corrupt_file: Path) -> None:
        """Un .dcm corrupto despachado desde ``read_image_file`` debe fallar con ImageReadError."""
        with pytest.raises(read_img.ImageReadError):
            read_img.read_image_file(corrupt_file)


class TestFuncionesInternas:
    """Pruebas directas sobre helpers privados de normalización."""

    def test_normalize_to_uint8_rango_constante(self) -> None:
        """Un arreglo de valor constante no debe producir división por cero."""
        valor_constante = 100
        arreglo = np.full((4, 4), valor_constante, dtype=np.uint16)
        resultado = read_img._normalize_to_uint8(arreglo)
        assert resultado.dtype == np.uint8
        assert np.all(resultado == VALOR_MINIMO_UINT8)

    def test_gray_to_rgb_forma_correcta(self) -> None:
        """La conversión gris->RGB debe triplicar el último eje."""
        lado = 4
        gris = np.arange(lado * lado, dtype=np.uint8).reshape(lado, lado)
        rgb = read_img._gray_to_rgb(gris)
        assert rgb.shape == (lado, lado, CANALES_RGB)
        assert np.array_equal(rgb[:, :, 0], gris)
