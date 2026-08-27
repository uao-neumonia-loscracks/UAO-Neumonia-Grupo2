"""Tests del módulo src/preprocess_img.py (capa Controlador)."""

from __future__ import annotations

import numpy as np
import pytest

import config
from preprocess_img import (
    apply_clahe,
    normalize,
    preprocess,
    resize_image,
    to_batch,
    to_grayscale,
)


def _imagen_rgb(alto: int, ancho: int) -> np.ndarray:
    """Genera una imagen RGB aleatoria uint8 de tamaño dado."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(alto, ancho, 3), dtype=np.uint8)


def _imagen_gris(alto: int, ancho: int) -> np.ndarray:
    """Genera una imagen en escala de grises aleatoria uint8 de tamaño dado."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(alto, ancho), dtype=np.uint8)


class TestResizeImage:
    """Tests de la función resize_image."""

    @pytest.mark.parametrize("alto,ancho", [(256, 256), (1024, 1024), (512, 700)])
    def test_redimensiona_a_tamano_config(self, alto, ancho):
        """Verifica que la salida siempre tenga el tamaño definido en config."""
        imagen = _imagen_rgb(alto, ancho)
        resultado = resize_image(imagen)
        assert resultado.shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3)
        assert resultado.dtype == np.uint8

    def test_pureza_no_modifica_entrada(self):
        """Verifica que la imagen original no se modifica al redimensionar."""
        imagen = _imagen_rgb(300, 400)
        copia = imagen.copy()
        resize_image(imagen)
        assert np.array_equal(imagen, copia)

    def test_error_entrada_1d(self):
        """Verifica que un arreglo 1D levanta ValueError."""
        with pytest.raises(ValueError):
            resize_image(np.zeros(10, dtype=np.uint8))

    def test_error_entrada_no_ndarray(self):
        """Verifica que una entrada que no es np.ndarray levanta ValueError."""
        with pytest.raises(ValueError):
            resize_image([[1, 2], [3, 4]])


class TestToGrayscale:
    """Tests de la función to_grayscale."""

    def test_convierte_rgb_a_gris(self):
        """Verifica la conversión de una imagen RGB a un solo canal."""
        imagen = _imagen_rgb(512, 512)
        resultado = to_grayscale(imagen)
        assert resultado.shape == (512, 512)
        assert resultado.dtype == np.uint8

    def test_idempotente_si_ya_es_2d(self):
        """Verifica que una imagen ya 2D se devuelve sin alterar (copia)."""
        imagen = _imagen_gris(256, 256)
        resultado = to_grayscale(imagen)
        assert np.array_equal(resultado, imagen)
        assert resultado is not imagen

    def test_pureza_no_modifica_entrada(self):
        """Verifica que la imagen original no se modifica."""
        imagen = _imagen_rgb(200, 200)
        copia = imagen.copy()
        to_grayscale(imagen)
        assert np.array_equal(imagen, copia)

    def test_error_dtype_float(self):
        """Verifica que una entrada en float levanta ValueError."""
        imagen = _imagen_rgb(64, 64).astype(np.float32)
        with pytest.raises(ValueError):
            to_grayscale(imagen)

    def test_error_forma_invalida(self):
        """Verifica que una entrada 1D levanta ValueError."""
        with pytest.raises(ValueError):
            to_grayscale(np.zeros(10, dtype=np.uint8))

    def test_error_canales_no_soportados(self):
        """Verifica que una imagen con canales distintos de 3 levanta ValueError."""
        imagen = np.zeros((10, 10, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            to_grayscale(imagen)


class TestApplyClahe:
    """Tests de la función apply_clahe."""

    def test_cambia_histograma_mantiene_shape_dtype(self):
        """Verifica que CLAHE cambia el histograma pero conserva shape/dtype."""
        imagen = np.full((512, 512), 128, dtype=np.uint8)
        imagen[100:400, 100:400] = 130
        resultado = apply_clahe(imagen)
        assert resultado.shape == imagen.shape
        assert resultado.dtype == np.uint8
        assert not np.isclose(float(resultado.std()), float(imagen.std()))

    def test_pureza_no_modifica_entrada(self):
        """Verifica que la imagen original no se modifica."""
        imagen = _imagen_gris(512, 512)
        copia = imagen.copy()
        apply_clahe(imagen)
        assert np.array_equal(imagen, copia)

    def test_error_entrada_3d(self):
        """Verifica que una entrada 3D levanta ValueError."""
        with pytest.raises(ValueError):
            apply_clahe(_imagen_rgb(64, 64))

    def test_error_dtype_no_uint8(self):
        """Verifica que una entrada en float levanta ValueError."""
        imagen = _imagen_gris(64, 64).astype(np.float32)
        with pytest.raises(ValueError):
            apply_clahe(imagen)


class TestNormalize:
    """Tests de la función normalize."""

    def test_uint8_a_float32_0_1(self):
        """Verifica la conversión de uint8 a float32 en rango [0,1]."""
        imagen = _imagen_gris(64, 64)
        resultado = normalize(imagen)
        assert resultado.dtype == np.float32
        assert resultado.min() >= 0.0
        assert resultado.max() <= 1.0

    def test_robusto_si_ya_esta_normalizada(self):
        """Verifica que una entrada ya normalizada no se divide de nuevo."""
        imagen = (_imagen_gris(64, 64).astype(np.float32)) / 255.0
        resultado = normalize(imagen)
        assert np.allclose(resultado, imagen)

    def test_pureza_no_modifica_entrada(self):
        """Verifica que la imagen original no se modifica."""
        imagen = _imagen_gris(64, 64)
        copia = imagen.copy()
        normalize(imagen)
        assert np.array_equal(imagen, copia)

    def test_error_entrada_no_ndarray(self):
        """Verifica que una entrada que no es np.ndarray levanta ValueError."""
        with pytest.raises(ValueError):
            normalize([1, 2, 3])


class TestToBatch:
    """Tests de la función to_batch."""

    def test_forma_y_dtype(self):
        """Verifica que la salida tenga forma (1,H,W,1) y dtype float32."""
        imagen = np.zeros((512, 512), dtype=np.float32)
        resultado = to_batch(imagen)
        assert resultado.shape == (1, 512, 512, 1)
        assert resultado.dtype == np.float32

    def test_pureza_no_modifica_entrada(self):
        """Verifica que la imagen original no se modifica."""
        imagen = np.random.default_rng(1).random((512, 512)).astype(np.float32)
        copia = imagen.copy()
        to_batch(imagen)
        assert np.array_equal(imagen, copia)

    def test_error_entrada_3d(self):
        """Verifica que una entrada 3D levanta ValueError."""
        with pytest.raises(ValueError):
            to_batch(_imagen_rgb(64, 64))


class TestPreprocessEndToEnd:
    """Tests de integración end-to-end de la función preprocess."""

    @pytest.mark.parametrize("alto,ancho", [(256, 256), (1024, 1024), (512, 700)])
    def test_pipeline_completo(self, alto, ancho):
        """Verifica shape, dtype y rango de salida para distintos tamaños de entrada."""
        imagen = _imagen_rgb(alto, ancho)
        resultado = preprocess(imagen)
        assert resultado.shape == (1, 512, 512, 1)
        assert resultado.dtype == np.float32
        assert resultado.min() >= 0.0
        assert resultado.max() <= 1.0

    def test_pureza_no_modifica_entrada_original(self):
        """Verifica que la imagen original no se modifica tras el pipeline completo."""
        imagen = _imagen_rgb(700, 512)
        copia = imagen.copy()
        preprocess(imagen)
        assert np.array_equal(imagen, copia)

    def test_error_entrada_1d(self):
        """Verifica que una entrada 1D levanta ValueError."""
        with pytest.raises(ValueError):
            preprocess(np.zeros(20, dtype=np.uint8))

    def test_error_entrada_no_ndarray(self):
        """Verifica que una entrada que no es np.ndarray levanta ValueError."""
        with pytest.raises(ValueError):
            preprocess("no es una imagen")
