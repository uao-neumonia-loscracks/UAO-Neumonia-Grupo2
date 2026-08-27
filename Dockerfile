# ============================================================================
# Dockerfile — UAO-Neumonia
# Multi-stage: "builder" resuelve dependencias con uv; "runtime" es la imagen
# final, mínima y sin herramientas de build.
# Nota: usa el builder clásico de Docker (sin BuildKit) por compatibilidad
# con entornos donde el plugin buildx no está disponible. Ver README para
# la variante con `RUN --mount=type=cache` una vez buildx esté instalado.
# ============================================================================

FROM python:3.13-slim-bookworm AS builder

# Binario estático de uv, copiado desde la imagen oficial de Astral (no se
# instala vía pip: el proyecto prohíbe pip por completo).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Capa cacheable de dependencias ---
# Solo copiamos los manifiestos primero: si src/ cambia pero no las
# dependencias, Docker reutiliza esta capa (tensorflow no se reinstala).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Ahora copiamos el código y reinstalamos el proyecto (rápido: las
# dependencias pesadas ya están cacheadas en la capa anterior).
COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev

# ============================================================================
FROM python:3.13-slim-bookworm AS runtime

# --- Paquetes de sistema mínimos (--no-install-recommends) ---
# tk           -> requerido por tkinter (la Vista, src/detector_neumonia.py)
#                 para renderizar widgets y el gestor de ventanas de Tk.
# libglib2.0-0 -> dependencia de runtime de OpenCV (glib) para operaciones
#                 básicas de imagen usadas por opencv-python-headless.
# libgomp1     -> OpenMP; TensorFlow lo requiere para paralelismo en CPU
#                 (kernels de convolución multihilo).
# libgl1       -> aunque usamos la variante "headless" de OpenCV, algunas
#                 rutas de carga dinámica de libGL siguen siendo resueltas
#                 en tiempo de import; sin esta lib el import falla en
#                 ciertas bases Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tk \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/models/conv_MLP_84.h5 \
    DISPLAY=:0 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/README.md /app/README.md
COPY scripts/ /app/scripts/
COPY tests/data/ /app/tests/data/

# El modelo NO se copia a la imagen:
#   1) Tamaño: el .h5 pesa varias decenas de MB y no debe inflar la imagen
#      base (contradice el objetivo de una imagen liviana y reproducible).
#   2) Licenciamiento: el binario entrenado no tiene una licencia de
#      redistribución confirmada para incluirlo en un artefacto Docker.
#   3) Gobernanza: está excluido por .gitignore y por decisión congelada del
#      plan maestro (MODEL_PATH resuelto en runtime, nunca commiteado ni
#      empaquetado). Se monta como volumen de solo lectura (ver README).
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

# Healthcheck liviano: solo verifica que los módulos del Controlador/Modelo
# sean importables (no ejecuta inferencia, para no depender del modelo real).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0,'src'); import read_img, preprocess_img, load_model, grad_cam, integrator" || exit 1

CMD ["python", "src/detector_neumonia.py"]


