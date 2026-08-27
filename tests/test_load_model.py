"""Tests de src/load_model.py (capa MODELO).

Cubre: manejo de errores sin modelo real, cálculo de SHA-256, búsqueda de la
última capa convolucional (con modelo dummy), comportamiento de la caché
`lru_cache`, y (marcado `requires_model`) la carga real de `conv_MLP_84.h5`.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pytest

import config
import load_model as lm

# Constantes de módulo para evitar magic values en comparaciones (ruff PLR2004).
_LONGITUD_SHA256_HEX = 64
_NUM_CLASES_ESPERADAS = 3


class TestModelNotFoundError:
    """Pruebas de `load_cnn_model` cuando el archivo no existe."""

    def test_lanza_model_not_found_error_con_ruta_inexistente(self, tmp_path: Path) -> None:
        """Debe lanzar ModelNotFoundError si la ruta no existe en disco."""
        lm.load_cnn_model.cache_clear()
        ruta_inexistente = tmp_path / "no_existe_conv_MLP_84.h5"

        with pytest.raises(lm.ModelNotFoundError) as excinfo:
            lm.load_cnn_model(model_path=str(ruta_inexistente))

        mensaje = str(excinfo.value)
        assert str(ruta_inexistente) in mensaje
        assert "MODEL_PATH" in mensaje
        assert "models/README.md" in mensaje

    def test_model_not_found_error_es_file_not_found_error(self, tmp_path: Path) -> None:
        """ModelNotFoundError debe seguir siendo un FileNotFoundError capturable."""
        lm.load_cnn_model.cache_clear()
        ruta_inexistente = tmp_path / "otro_no_existe.h5"

        with pytest.raises(FileNotFoundError):
            lm.load_cnn_model(model_path=str(ruta_inexistente))


class TestComputeSha256:
    """Pruebas de `compute_sha256` contra un archivo temporal de contenido conocido."""

    def test_compute_sha256_contenido_conocido(self, tmp_path: Path) -> None:
        """El hash calculado debe coincidir con el hash de referencia de hashlib."""
        archivo = tmp_path / "contenido_conocido.bin"
        contenido = b"UAO-Neumonia - contenido de prueba fijo para SHA-256"
        archivo.write_bytes(contenido)

        hash_esperado = hashlib.sha256(contenido).hexdigest()
        hash_real = lm.compute_sha256(archivo)

        assert hash_real == hash_esperado
        assert len(hash_real) == _LONGITUD_SHA256_HEX

    def test_compute_sha256_archivo_inexistente(self, tmp_path: Path) -> None:
        """Debe lanzar FileNotFoundError si el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            lm.compute_sha256(tmp_path / "no_existe.bin")


class TestVerifyModelIntegrity:
    """Pruebas de `verify_model_integrity`."""

    def test_sin_hash_esperado_devuelve_true_si_existe(self, tmp_path: Path) -> None:
        """Sin hash de referencia, solo valida que el archivo sea legible."""
        archivo = tmp_path / "modelo_falso.h5"
        archivo.write_bytes(b"contenido binario de prueba")

        assert lm.verify_model_integrity(archivo) is True

    def test_hash_esperado_coincide(self, tmp_path: Path) -> None:
        """Con hash de referencia correcto, debe devolver True."""
        archivo = tmp_path / "modelo_falso.h5"
        contenido = b"otro contenido de prueba"
        archivo.write_bytes(contenido)
        hash_correcto = hashlib.sha256(contenido).hexdigest()

        assert lm.verify_model_integrity(archivo, expected_sha256=hash_correcto) is True

    def test_hash_esperado_no_coincide(self, tmp_path: Path) -> None:
        """Con hash de referencia incorrecto, debe devolver False."""
        archivo = tmp_path / "modelo_falso.h5"
        archivo.write_bytes(b"contenido cualquiera")

        assert (
            lm.verify_model_integrity(archivo, expected_sha256="0" * _LONGITUD_SHA256_HEX) is False
        )

    def test_archivo_inexistente_devuelve_false(self, tmp_path: Path) -> None:
        """Si el archivo no existe, debe devolver False sin lanzar excepción."""
        assert lm.verify_model_integrity(tmp_path / "fantasma.h5") is False


class TestGetLastConvLayerName:
    """Pruebas de `get_last_conv_layer_name` con un modelo Keras dummy."""

    @pytest.fixture()
    def modelo_dummy(self):
        """Construye un Sequential pequeño: 2 Conv2D + Dense (sin entrenar)."""
        modelo = lm.keras.Sequential(
            [
                lm.keras.layers.Input(shape=(16, 16, 1)),
                lm.keras.layers.Conv2D(4, (3, 3), name="conv_primera", padding="same"),
                lm.keras.layers.Conv2D(8, (3, 3), name="conv_ultima", padding="same"),
                lm.keras.layers.Flatten(),
                lm.keras.layers.Dense(3, name="dense_salida"),
            ]
        )
        return modelo

    def test_encuentra_la_ultima_capa_conv2d(self, modelo_dummy) -> None:
        """Debe devolver el nombre de la última Conv2D, no la primera."""
        nombre = lm.get_last_conv_layer_name(modelo_dummy)
        assert nombre == "conv_ultima"

    def test_lanza_value_error_sin_capas_conv(self) -> None:
        """Debe lanzar ValueError si el modelo no tiene ninguna capa convolucional."""
        modelo_sin_conv = lm.keras.Sequential(
            [
                lm.keras.layers.Input(shape=(10,)),
                lm.keras.layers.Dense(5, name="dense_a"),
                lm.keras.layers.Dense(2, name="dense_b"),
            ]
        )

        with pytest.raises(ValueError, match="Conv2D"):
            lm.get_last_conv_layer_name(modelo_sin_conv)


class TestLruCacheIdentity:
    """Prueba de que `load_cnn_model` cachea y devuelve la MISMA instancia."""

    def test_llamadas_repetidas_devuelven_la_misma_instancia(self, tmp_path: Path) -> None:
        """Dos llamadas con la misma ruta deben devolver el mismo objeto (a is b).

        Nota: `tf_keras` emite un `UserWarning` de "formato HDF5 legacy" al
        GUARDAR (no al cargar) un modelo con `.save(archivo.h5)`. Como
        `pyproject.toml` define `filterwarnings = ["error", ...]`, ese aviso
        se escalaría a excepción y rompería este test, que solo usa `.save()`
        para preparar un archivo `.h5` de prueba (no es el camino de
        `load_cnn_model`, que nunca guarda modelos). Se silencia de forma
        LOCALIZADA a este bloque, no globalmente, porque es ruido propio de
        la API de guardado de tf_keras y no de nuestro código de carga.
        """
        lm.load_cnn_model.cache_clear()

        modelo_dummy = lm.keras.Sequential(
            [
                lm.keras.layers.Input(shape=(8, 8, 1)),
                lm.keras.layers.Conv2D(2, (3, 3), name="conv_unica", padding="same"),
                lm.keras.layers.Flatten(),
                lm.keras.layers.Dense(3, name="dense_salida"),
            ]
        )
        ruta_modelo_dummy = tmp_path / "dummy_cache.h5"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            modelo_dummy.save(ruta_modelo_dummy)

        modelo_a = lm.load_cnn_model(model_path=str(ruta_modelo_dummy))
        modelo_b = lm.load_cnn_model(model_path=str(ruta_modelo_dummy))

        assert modelo_a is modelo_b

        lm.load_cnn_model.cache_clear()


class TestLoadCnnModelReal:
    """Prueba con el archivo real `conv_MLP_84.h5` (requiere el binario)."""

    @pytest.mark.requires_model
    @pytest.mark.skipif(
        not config.get_model_path().is_file(),
        reason="conv_MLP_84.h5 no está presente; ver models/README.md",
    )
    def test_carga_modelo_real_y_verifica_shapes(self) -> None:
        """Carga el modelo real y valida input_shape/output_shape esperados."""
        lm.load_cnn_model.cache_clear()

        modelo = lm.load_cnn_model()

        assert modelo.input_shape == (None, 512, 512, 1)
        assert modelo.output_shape[-1] == _NUM_CLASES_ESPERADAS

        nombre_capa = lm.get_last_conv_layer_name(modelo)
        assert isinstance(nombre_capa, str)
        assert len(nombre_capa) > 0
