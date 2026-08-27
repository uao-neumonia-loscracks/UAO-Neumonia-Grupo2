"""Fixtures compartidas para las pruebas de UAO-Neumonia.

Genera archivos sintéticos (DICOM válidos, JPG, PNG y archivos corruptos) en
directorios temporales de pytest, sin depender de descargas externas ni de
datos reales de pacientes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

ROWS = 512
COLUMNS = 512


def _patron_reconocible() -> np.ndarray:
    """Genera un arreglo uint16 (512, 512) con un patrón de gradiente reconocible.

    Returns:
        np.ndarray: Arreglo 2D uint16 con valores entre 0 y 4095 (12 bits),
        variando a lo largo del eje horizontal, útil para verificar
        normalización e inversión de intensidades en las pruebas.
    """
    fila = np.linspace(0, 4095, COLUMNS, dtype=np.float64)
    patron = np.tile(fila, (ROWS, 1)).astype(np.uint16)
    return patron


def _crear_dataset_dicom(pixel_array: np.ndarray, photometric_interpretation: str) -> FileDataset:
    """Construye un ``FileDataset`` DICOM válido en memoria para pruebas.

    La sintaxis de transferencia se define únicamente mediante
    ``file_meta.TransferSyntaxUID``; no se asignan los atributos
    ``is_little_endian``/``is_implicit_VR`` en el dataset porque están
    deprecados en pydicom >=3.x cuando ya existe un ``TransferSyntaxUID``
    explícito (asignarlos lanza ``DeprecationWarning``, que este proyecto
    trata como error en pytest).

    Args:
        pixel_array: Arreglo 2D uint16 con los datos de píxeles a incrustar.
        photometric_interpretation: Valor a asignar a
            ``PhotometricInterpretation`` (p. ej. "MONOCHROME1" o
            "MONOCHROME2").

    Returns:
        FileDataset: Dataset DICOM completo, listo para guardarse con
        ``save_as``.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(
        filename_or_obj=None,
        dataset={},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "OT"
    dataset.PatientName = "Test^Sintetico"
    dataset.PatientID = "TEST0001"

    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.Rows = pixel_array.shape[0]
    dataset.Columns = pixel_array.shape[1]
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = "1"
    dataset.RescaleIntercept = "0"
    dataset.PixelData = pixel_array.astype(np.uint16).tobytes()

    return dataset


@pytest.fixture
def synthetic_dicom(tmp_path: Path) -> Path:
    """Crea un archivo DICOM válido MONOCHROME2 con un patrón reconocible.

    Args:
        tmp_path: Directorio temporal provisto por pytest.

    Returns:
        Path: Ruta al archivo ``sintetico_mono2.dcm`` generado.
    """
    dataset = _crear_dataset_dicom(_patron_reconocible(), "MONOCHROME2")
    ruta = tmp_path / "sintetico_mono2.dcm"
    dataset.save_as(str(ruta), enforce_file_format=True)
    return ruta


@pytest.fixture
def synthetic_dicom_monochrome1(tmp_path: Path) -> Path:
    """Crea un archivo DICOM válido MONOCHROME1 con el mismo patrón reconocible.

    Args:
        tmp_path: Directorio temporal provisto por pytest.

    Returns:
        Path: Ruta al archivo ``sintetico_mono1.dcm`` generado.
    """
    dataset = _crear_dataset_dicom(_patron_reconocible(), "MONOCHROME1")
    ruta = tmp_path / "sintetico_mono1.dcm"
    dataset.save_as(str(ruta), enforce_file_format=True)
    return ruta


@pytest.fixture
def synthetic_jpg(tmp_path: Path) -> Path:
    """Crea una imagen JPG sintética RGB de 64x64 píxeles.

    Args:
        tmp_path: Directorio temporal provisto por pytest.

    Returns:
        Path: Ruta al archivo ``sintetica.jpg`` generado.
    """
    arreglo = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    ruta = tmp_path / "sintetica.jpg"
    Image.fromarray(arreglo, mode="RGB").save(ruta, format="JPEG")
    return ruta


@pytest.fixture
def synthetic_png(tmp_path: Path) -> Path:
    """Crea una imagen PNG sintética en escala de grises de 64x64 píxeles.

    Args:
        tmp_path: Directorio temporal provisto por pytest.

    Returns:
        Path: Ruta al archivo ``sintetica.png`` generado.
    """
    arreglo = np.random.default_rng(7).integers(0, 256, (64, 64), dtype=np.uint8)
    ruta = tmp_path / "sintetica.png"
    Image.fromarray(arreglo, mode="L").save(ruta, format="PNG")
    return ruta


@pytest.fixture
def corrupt_file(tmp_path: Path) -> Path:
    """Crea un archivo con extensión ``.dcm`` pero contenido inválido/corrupto.

    Args:
        tmp_path: Directorio temporal provisto por pytest.

    Returns:
        Path: Ruta al archivo ``corrupto.dcm`` con bytes arbitrarios que no
        constituyen un DICOM válido.
    """
    ruta = tmp_path / "corrupto.dcm"
    ruta.write_bytes(b"esto-no-es-un-dicom-valido" * 10)
    return ruta
