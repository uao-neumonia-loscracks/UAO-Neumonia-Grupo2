## 5. Contratos de módulos (API congelada)

Ningún hilo puede cambiar una firma sin registrar un ADR. Guardar como `docs/CONTRATOS.md`.
 
### 5.1 `src/config.py` — constantes centralizadas
 
```python
IMAGE_SIZE: tuple[int, int] = (512, 512)
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (4, 4)
CLASS_LABELS: dict[int, str] = {0: "bacteriana", 1: "normal", 2: "viral"}
DEFAULT_MODEL_FILENAME: str = "conv_MLP_84.h5"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".dcm", ".dicom", ".jpg", ".jpeg", ".png")
 
def get_model_path() -> Path: ...
def get_project_root() -> Path: ...
```
 
### 5.2 `src/read_img.py` — CONTROLADOR
 
```python
def read_dicom_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Lee un archivo DICOM y devuelve (array RGB uint8, imagen PIL para la GUI)."""
 
def read_jpg_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Lee JPG/PNG y devuelve (array RGB uint8, imagen PIL para la GUI)."""
 
def read_image_file(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """Despachador por extensión. Lanza UnsupportedFormatError si no aplica."""
 
class UnsupportedFormatError(ValueError): ...
class ImageReadError(RuntimeError): ...
```
 
**Invariantes verificables:** salida `ndim == 3`, `shape[2] == 3`, `dtype == np.uint8`, `0 <= min <= max <= 255`.
 
### 5.3 `src/preprocess_img.py` — CONTROLADOR
 
```python
def resize_image(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray: ...
def to_grayscale(image: np.ndarray) -> np.ndarray: ...
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple[int, int] = (4, 4)) -> np.ndarray: ...
def normalize(image: np.ndarray) -> np.ndarray: ...
def to_batch(image: np.ndarray) -> np.ndarray: ...
def preprocess(image: np.ndarray) -> np.ndarray:
    """Pipeline completo: resize → gris → CLAHE → normalización → batch (1,512,512,1)."""
```
 
**Invariantes:** `preprocess(x).shape == (1,512,512,1)`, `dtype == float32`, `0.0 <= x <= 1.0`.
Cada función es **pura** (no muta la entrada) — hay test que lo comprueba.
 
### 5.4 `src/load_model.py` — MODELO
 
```python
@lru_cache(maxsize=1)
def load_cnn_model(model_path: str | None = None) -> "keras.Model":
    """Carga (y cachea) la CNN desde el .h5. Lanza ModelNotFoundError si no existe."""
 
def get_last_conv_layer_name(model: "keras.Model") -> str:
    """Devuelve el nombre de la última capa convolucional (objetivo de Grad-CAM)."""
 
def compute_sha256(path: Path) -> str: ...
def verify_model_integrity(path: Path, expected_sha256: str | None = None) -> bool: ...
def model_summary_text(model) -> str:   # para el Model Card y la sustentación
class ModelNotFoundError(FileNotFoundError): ...
```
 
### 5.5 `src/grad_cam.py` — MODELO
 
```python
def compute_gradcam_heatmap(batch: np.ndarray, class_index: int | None = None,
                            layer_name: str | None = None) -> tuple[np.ndarray, int, float]:
    """Devuelve (heatmap 512x512 float32 [0,1], índice de clase, probabilidad)."""
 
def overlay_heatmap(heatmap: np.ndarray, original_rgb: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Superpone el mapa de calor (JET) sobre la radiografía. Devuelve RGB uint8."""
 
def grad_cam(image_rgb: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Fachada: imagen cruda → (overlay RGB uint8, class_index, probabilidad)."""
```
 
### 5.6 `src/integrator.py` — CONTROLADOR (orquestador)
 
```python
@dataclass(frozen=True)
class PredictionResult:
    label: str            # "bacteriana" | "normal" | "viral"
    probability: float    # 0.0 – 100.0 (porcentaje)
    class_index: int      # 0 | 1 | 2
    heatmap: np.ndarray   # overlay RGB uint8 (512,512,3)
 
def predict(image_rgb: np.ndarray) -> PredictionResult:
    """Única puerta de entrada de la Vista al sistema. Orquesta preprocess → modelo → Grad-CAM."""
```
 
> ⚠️ **Regla de acoplamiento:** `detector_neumonia.py` solo puede hacer
> `from integrator import predict, PredictionResult` y `from read_img import read_image_file`.
> Nada más. Esto lo verifica automáticamente `tests/test_architecture.py`.
 
### 5.7 `src/detector_neumonia.py` — VISTA
 
```python
class PneumoniaDetectorApp(tk.Tk):
    def __init__(self) -> None: ...
    def on_load_image(self) -> None: ...
    def on_predict(self) -> None: ...
    def on_save_csv(self) -> None: ...
    def on_export_pdf(self) -> None: ...
    def on_clear(self) -> None: ...
 
def main() -> None:
    """Punto de entrada de la aplicación."""
```
 
---
