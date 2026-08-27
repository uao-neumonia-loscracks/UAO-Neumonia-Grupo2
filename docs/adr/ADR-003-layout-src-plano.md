# ADR-003: Layout `src/` plano vs paquete anidado

## Contexto

La rúbrica del curso exige archivos con nombre literal (`src/read_img.py`,
`src/detector_neumonia.py`, etc.), importables tal cual (`import read_img`), sin un
paquete Python anidado de por medio (por ejemplo, `src/uao_neumonia/read_img.py` con
`import uao_neumonia.read_img`). Esto entra en tensión con la convención habitual de
empaquetar código Python bajo un paquete con `__init__.py`.

## Decisión

Mantener `src/` **plano** (sin subpaquete), usando `setuptools` con
`package-dir = {"" = "src"}` y una lista explícita de `py-modules` en `pyproject.toml`
(`config`, `read_img`, `preprocess_img`, `load_model`, `grad_cam`, `integrator`,
`detector_neumonia`) en lugar de `find_packages()`.

## Alternativas consideradas

- **Paquete anidado `src/uao_neumonia/*.py`** (descartado): más idiomático para
  distribución en PyPI, pero rompe el requisito literal de nombres de archivo de la
  rúbrica y obliga a imports con prefijo (`from uao_neumonia import integrator`) en toda
  la GUI y los tests.
- **`find_packages()` automático** (descartado): requiere que exista un directorio de
  paquete real (con `__init__.py`), lo cual reintroduce el mismo problema que la opción
  anterior.

## Consecuencias

- Los imports en todo el código son planos: `from integrator import predict,
  PredictionResult`, `from read_img import read_image_file`.
- `pyproject.toml` debe listar cada módulo nuevo explícitamente en `py-modules`; añadir un
  archivo a `src/` sin actualizar esa lista lo dejará fuera del paquete instalado (aunque
  sí funcionará en modo desarrollo vía `uv run` desde la raíz, por el `PYTHONPATH`
  implícito de `src`).
- `tests/test_architecture.py` puede verificar imports por nombre de módulo simple, sin
  necesidad de resolver rutas de paquete anidadas.

## Estado

**Aceptado.**