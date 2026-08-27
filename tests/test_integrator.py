"""Tests para src/integrator.py (capa Controlador — orquestador)."""

import dataclasses
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import integrator
import read_img
from config import CLASS_LABELS
from integrator import PredictionError, PredictionResult, predict

_FORMA_HEATMAP = (512, 512, 3)
_FORMA_IMAGEN = (512, 512, 3)
_PROBABILIDAD_VALIDA = 87.65
_PROBABILIDAD_MINIMA_TEST = 0.0
_PROBABILIDAD_MAXIMA_TEST = 100.0


def _heatmap_valido() -> np.ndarray:
    """Genera un overlay Grad-CAM sintético válido para los mocks."""
    return np.zeros(_FORMA_HEATMAP, dtype=np.uint8)


def _imagen_valida() -> np.ndarray:
    """Genera una imagen RGB uint8 sintética válida para los tests."""
    return np.zeros(_FORMA_IMAGEN, dtype=np.uint8)


class TestPredictConMocks:
    """Verifica la orquestación de `predict` sin cargar el modelo real."""

    @pytest.mark.parametrize("class_index", sorted(CLASS_LABELS))
    def test_predict_mapea_indice_a_etiqueta(self, mocker, class_index: int) -> None:
        """Cada índice de CLASS_LABELS (0, 1, 2) se mapea a su etiqueta correcta."""
        heatmap = _heatmap_valido()
        mock_grad_cam = mocker.patch(
            "integrator.grad_cam",
            return_value=(heatmap, class_index, _PROBABILIDAD_VALIDA),
        )

        resultado = predict(_imagen_valida())

        mock_grad_cam.assert_called_once()
        assert resultado.class_index == class_index
        assert resultado.label == CLASS_LABELS[class_index]
        assert resultado.probability == pytest.approx(_PROBABILIDAD_VALIDA)
        assert resultado.heatmap.shape == _FORMA_HEATMAP

    def test_predict_no_importa_preprocess_directamente(self) -> None:
        """Confirma la decisión D6.1: integrator no expone/usa preprocess propio."""
        assert not hasattr(integrator, "preprocess")

    def test_predict_indice_desconocido_lanza_prediction_error(self, mocker) -> None:
        """Un class_index fuera de CLASS_LABELS se envuelve en PredictionError."""
        mocker.patch(
            "integrator.grad_cam",
            return_value=(_heatmap_valido(), 99, _PROBABILIDAD_VALIDA),
        )
        with pytest.raises(PredictionError):
            predict(_imagen_valida())

    def test_predict_fallo_grad_cam_se_envuelve_en_prediction_error(self, mocker) -> None:
        """Un fallo interno de grad_cam() nunca se filtra sin envolver."""
        mocker.patch(
            "integrator.grad_cam",
            side_effect=RuntimeError("fallo interno de TensorFlow"),
        )
        with pytest.raises(PredictionError) as exc_info:
            predict(_imagen_valida())
        assert exc_info.value.causa is not None

    def test_predict_registra_log_info(self, mocker, caplog: pytest.LogCaptureFixture) -> None:
        """predict() registra un log INFO con la etiqueta y la probabilidad."""
        mocker.patch(
            "integrator.grad_cam",
            return_value=(_heatmap_valido(), 1, _PROBABILIDAD_VALIDA),
        )
        with caplog.at_level(logging.INFO, logger="integrator"):
            predict(_imagen_valida())
        assert any("normal" in registro.message for registro in caplog.records)


class TestValidacionEntrada:
    """Verifica que predict() rechace entradas inválidas con ValueError."""

    def test_no_ndarray_lanza_value_error(self) -> None:
        """Una entrada que no es np.ndarray debe rechazarse."""
        with pytest.raises(ValueError):
            predict([[1, 2, 3]])  # type: ignore[arg-type]

    def test_ndim_incorrecto_lanza_value_error(self) -> None:
        """Una imagen sin canal (2D) debe rechazarse."""
        with pytest.raises(ValueError):
            predict(np.zeros((512, 512), dtype=np.uint8))

    def test_canales_incorrectos_lanza_value_error(self) -> None:
        """Una imagen con un número de canales distinto de 3 debe rechazarse."""
        with pytest.raises(ValueError):
            predict(np.zeros((512, 512, 4), dtype=np.uint8))

    def test_dtype_incorrecto_lanza_value_error(self) -> None:
        """Una imagen con dtype distinto de uint8 debe rechazarse."""
        with pytest.raises(ValueError):
            predict(np.zeros((512, 512, 3), dtype=np.float32))


class TestPredictionResultInmutable:
    """Verifica inmutabilidad y validación de invariantes de PredictionResult."""

    def test_asignar_campo_lanza_frozen_instance_error(self) -> None:
        """Reasignar un campo de un PredictionResult ya construido debe fallar."""
        resultado = PredictionResult(
            label="normal",
            probability=90.0,
            class_index=1,
            heatmap=_heatmap_valido(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            resultado.label = "bacteriana"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"probability": -1.0},
            {"probability": 101.0},
            {"class_index": 7},
            {"heatmap": np.zeros((10, 10, 3), dtype=np.uint8)},
            {"heatmap": np.zeros((512, 512, 3), dtype=np.float32)},
        ],
    )
    def test_post_init_valida_rangos(self, kwargs: dict) -> None:
        """Cada combinación inválida de campos debe ser rechazada por __post_init__."""
        base = {
            "label": "normal",
            "probability": 90.0,
            "class_index": 1,
            "heatmap": _heatmap_valido(),
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            PredictionResult(**base)


class TestCasosBordeAdicionales:
    """Casos borde no cubiertos por la batería original del prototipo."""

    @pytest.mark.parametrize("probabilidad", [_PROBABILIDAD_MINIMA_TEST, _PROBABILIDAD_MAXIMA_TEST])
    def test_probabilidad_en_los_limites_exactos_se_acepta(self, probabilidad: float) -> None:
        """Los extremos 0.0 y 100.0 son válidos: el rango es cerrado, no abierto.

        La batería original solo comprobaba que -1.0 y 101.0 fueran rechazados,
        lo que dejaba pasar un off-by-one si `__post_init__` usara `<` en vez
        de `<=`. Este test fija el contrato del rango inclusivo.
        """
        resultado = PredictionResult(
            label="normal",
            probability=probabilidad,
            class_index=1,
            heatmap=_heatmap_valido(),
        )
        assert resultado.probability == pytest.approx(probabilidad)

    def test_tipos_numpy_del_modelo_se_normalizan_a_tipos_python(self, mocker) -> None:
        """`class_index` y `probability` salen como int/float nativos de Python.

        En la ejecución real `grad_cam()` deriva el índice de `np.argmax()`
        (`np.int64`) y la probabilidad de un tensor de TensorFlow
        (`np.float32`). Si `predict` no los convirtiera, la Vista terminaría
        serializando tipos numpy al exportar a CSV o PDF. Este test cubre esa
        frontera, que los mocks con `int`/`float` puros no ejercitaban.
        """
        mocker.patch(
            "integrator.grad_cam",
            return_value=(_heatmap_valido(), np.int64(2), np.float32(88.5)),
        )

        resultado = predict(_imagen_valida())

        assert resultado.label == "viral"
        assert type(resultado.class_index) is int
        assert type(resultado.probability) is float

    def test_imagen_de_tamano_arbitrario_se_acepta(self, mocker) -> None:
        """`predict` no exige 512x512: acepta cualquier (H, W, 3) uint8.

        `read_img.read_image_file` devuelve la radiografía en su resolución
        original y es `grad_cam()` quien la redimensiona. Fijar aquí un tamaño
        rompería ese reparto de responsabilidades.
        """
        mocker.patch(
            "integrator.grad_cam",
            return_value=(_heatmap_valido(), 1, _PROBABILIDAD_VALIDA),
        )

        resultado = predict(np.zeros((256, 300, 3), dtype=np.uint8))

        assert resultado.label == "normal"


@pytest.mark.requires_model
class TestPredictEndToEndModeloReal:
    """Test end-to-end con el modelo real conv_MLP_84.h5 (requiere MODEL_PATH)."""

    def test_predict_con_imagen_real(self) -> None:
        """predict() debe producir un resultado válido sobre un DICOM real."""
        ruta = os.environ.get("UAO_TEST_DICOM_PATH")
        if not ruta or not Path(ruta).exists():
            pytest.skip("Define UAO_TEST_DICOM_PATH con una ruta DICOM real para este test.")

        imagen_rgb, _ = read_img.read_image_file(ruta)
        resultado = predict(imagen_rgb)

        assert isinstance(resultado, PredictionResult)
        assert resultado.label in CLASS_LABELS.values()
        assert _PROBABILIDAD_MINIMA_TEST <= resultado.probability <= _PROBABILIDAD_MAXIMA_TEST
        assert resultado.heatmap.shape == _FORMA_HEATMAP
        assert resultado.heatmap.dtype == np.uint8
