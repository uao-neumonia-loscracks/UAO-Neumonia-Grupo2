## 📝 Descripción
<!-- Qué cambia y por qué. -->

## 🔧 Tipo de cambio
- [ ] Nueva funcionalidad
- [ ] Corrección de bug
- [ ] Refactor sin cambio de comportamiento
- [ ] Documentación
- [ ] Infraestructura / CI / tooling

## 🏗️ Capa MVC afectada
- [ ] Modelo (`load_model.py`, `grad_cam.py`)
- [ ] Controlador (`read_img.py`, `preprocess_img.py`, `integrator.py`)
- [ ] Vista (`detector_neumonia.py`)
- [ ] Transversal (`config.py`, tests, docs, CI)

## 🔗 Issue relacionado
Closes #

## ✅ Checklist
- [ ] `uv run ruff check .` sin findings
- [ ] `uv run ruff format --check .` sin cambios pendientes
- [ ] `uv run pytest` en verde (incluye `tests/test_architecture.py`)
- [ ] Docstrings Google en español en todas las funciones/clases nuevas
- [ ] Cero warnings nuevos (`filterwarnings = ["error", ...]` no falla)
- [ ] Sin credenciales, tokens ni rutas absolutas locales
- [ ] Sin binarios commiteados (`conv_MLP_84.h5` u otros `.h5`)

## 📎 Evidencia
<!-- Salida de comandos, capturas de la GUI o del heatmap. -->

## 🧪 Cómo probarlo paso a paso
1. …
2. …

## 👤 Revisor asignado
@…
