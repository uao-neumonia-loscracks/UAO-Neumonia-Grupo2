"""Genera un archivo DICOM sintético para pruebas y smoke tests.

Crea `tests/data/sample_synthetic.dcm`: una imagen de rayos X sintética
(ruido aleatorio reproducible, 512x512, escala de grises) con los metadatos
mínimos válidos para ser leída por `pydicom` y por `src/read_img.py`.

No contiene datos de pacientes reales: es 100% sintética, generada con una
semilla fija para reproducibilidad.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "data" / "sample_synthetic.dcm"
IMAGE_SIZE = (512, 512)


def build_synthetic_dicom(size: tuple[int, int] = IMAGE_SIZE, seed: int = 42) -> FileDataset:
    """Construye un dataset DICOM sintético en memoria.

    Args:
        size: dimensiones (filas, columnas) de la imagen sintética.
        seed: semilla del generador aleatorio, para reproducibilidad.

    Returns:
        Un `FileDataset` de pydicom listo para escribirse a disco.
    """
    rng = np.random.default_rng(seed)
    pixel_array = rng.integers(0, 256, size=size, dtype=np.uint8)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(OUTPUT_PATH), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "SYNTHETIC^TEST"
    ds.PatientID = "SYNTH0001"
    ds.Modality = "CR"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()

    ds.Rows, ds.Columns = size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()

    return ds


def main() -> None:
    """Genera y guarda el archivo DICOM sintético en tests/data/."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_synthetic_dicom()
    dataset.save_as(
        OUTPUT_PATH,
        enforce_file_format=True,
        little_endian=True,
        implicit_vr=False,
    )
    logger.info("DICOM sintético generado en: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
