"""Tests para src/grad_cam.py (capa MODELO).

Verifica el algoritmo Grad-CAM con una CNN dummy entrenable construida en el
propio test, y una prueba end-to-end (marcada ``requires_model``) contra el
modelo real ``conv_MLP_84.h5``.
"""

from __future__ import annotations

import numpy as np
import pytest

import grad_cam
from grad_cam import keras

_TAM_IMAGEN = 512
_CANALES_CONV = 4
_NUM_CLASES = 3
_SEMILLA = 42
_PROB_MAXIMA = 100.0


def _construir_modelo_dummy() -> keras.Model:
    """Construye una CNN dummy (Conv2D + GAP + Dense) para pruebas rápidas."""
    keras.utils.set_random_seed(_SEMILLA)
    entradas = keras.Input(shape=(_TAM_IMAGEN, _TAM_IMAGEN, 1))
    x = keras.layers.Conv2D(_CANALES_CONV, 3, padding="same", name="conv_dummy")(entradas)
    x = keras.layers.GlobalAveragePooling2D()(x)
    salidas = keras.layers.Dense(_NUM_CLASES, activation="softmax")(x)
    return keras.Model(inputs=entradas, outputs=salidas)


@pytest.fixture(scope="module")
def modelo_dummy() -> keras.Model:
    """Fixture con la CNN dummy compartida entre tests de este módulo."""
    return _construir_modelo_dummy()


@pytest.fixture
def batch_aleatorio() -> np.ndarray:
    """Batch sintético (1, 512, 512, 1) float32 en [0, 1]."""
    rng = np.random.default_rng(_SEMILLA)
    return rng.random((1, _TAM_IMAGEN, _TAM_IMAGEN, 1)).astype(np.float32)


class TestComputeGradcamHeatmap:
    """Pruebas de compute_gradcam_heatmap con modelo dummy."""

    def test_shape_dtype_rango(self, monkeypatch, modelo_dummy, batch_aleatorio):
        """El heatmap debe ser (512,512) float32 en [0,1] y no degenerado."""
        monkeypatch.setattr(grad_cam, "load_cnn_model", lambda: modelo_dummy)

        heatmap, indice_clase, probabilidad = grad_cam.compute_gradcam_heatmap(batch_aleatorio)

        assert heatmap.shape == (_TAM_IMAGEN, _TAM_IMAGEN)
        assert heatmap.dtype == np.float32
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        assert np.any(heatmap > 0.0), "El heatmap no debe ser todo ceros"
        assert 0 <= indice_clase < _NUM_CLASES
        assert 0.0 <= probabilidad <= _PROB_MAXIMA

    def test_determinismo(self, monkeypatch, modelo_dummy, batch_aleatorio):
        """Dos llamadas seguidas con el mismo batch dan el mismo heatmap."""
        monkeypatch.setattr(grad_cam, "load_cnn_model", lambda: modelo_dummy)

        heatmap_1, clase_1, prob_1 = grad_cam.compute_gradcam_heatmap(
            batch_aleatorio, class_index=0
        )
        heatmap_2, clase_2, prob_2 = grad_cam.compute_gradcam_heatmap(
            batch_aleatorio, class_index=0
        )

        assert np.array_equal(heatmap_1, heatmap_2)
        assert clase_1 == clase_2
        assert prob_1 == pytest.approx(prob_2)


class TestOverlayHeatmap:
    """Pruebas de overlay_heatmap con arrays sintéticos."""

    @pytest.fixture
    def heatmap_sintetico(self) -> np.ndarray:
        """Heatmap sintético con gradiente diagonal en [0,1]."""
        eje = np.linspace(0.0, 1.0, _TAM_IMAGEN, dtype=np.float32)
        return np.outer(eje, eje)

    @pytest.fixture
    def imagen_original(self) -> np.ndarray:
        """Imagen RGB sintética uint8 (512,512,3) de valor constante gris."""
        return np.full((_TAM_IMAGEN, _TAM_IMAGEN, 3), 128, dtype=np.uint8)

    def test_shape_dtype_y_difiere_de_original(self, heatmap_sintetico, imagen_original):
        """El overlay debe ser uint8 (512,512,3) y distinto de la original."""
        overlay = grad_cam.overlay_heatmap(heatmap_sintetico, imagen_original, alpha=0.4)

        assert overlay.shape == (_TAM_IMAGEN, _TAM_IMAGEN, 3)
        assert overlay.dtype == np.uint8
        assert not np.array_equal(overlay, imagen_original)

    def test_alpha_cero_igual_a_original(self, heatmap_sintetico, imagen_original):
        """Con alpha=0 el overlay debe ser idéntico a la original."""
        overlay = grad_cam.overlay_heatmap(heatmap_sintetico, imagen_original, alpha=0.0)

        assert np.array_equal(overlay, imagen_original)


@pytest.mark.requires_model
class TestGradCamEndToEnd:
    """Prueba end-to-end con el modelo real conv_MLP_84.h5."""

    def test_grad_cam_facade_con_modelo_real(self):
        """La fachada grad_cam() produce un overlay coherente con el modelo real."""
        rng = np.random.default_rng(_SEMILLA)
        imagen_rgb = rng.integers(0, 256, size=(_TAM_IMAGEN, _TAM_IMAGEN, 3), dtype=np.uint8)

        overlay, indice_clase, probabilidad = grad_cam.grad_cam(imagen_rgb)

        assert overlay.shape == (_TAM_IMAGEN, _TAM_IMAGEN, 3)
        assert overlay.dtype == np.uint8
        assert 0 <= indice_clase < _NUM_CLASES
        assert 0.0 <= probabilidad <= _PROB_MAXIMA
