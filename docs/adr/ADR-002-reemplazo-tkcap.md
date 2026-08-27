# ADR-002: Reemplazo de tkcap + img2pdf por reportlab

## Contexto

El flujo original de exportación a PDF de la GUI dependía de `tkcap` (captura de la
ventana Tkinter) e `img2pdf` (conversión de esa captura a PDF). Ambas librerías están
abandonadas y no son compatibles con Python 3.13/Tkinter moderno, lo que las convierte en
un bloqueante para el requisito de "cero código deprecado" del proyecto.

## Decisión

Generar el PDF **directamente con `reportlab`**, construyendo el documento
programáticamente (texto de la predicción + imagen del overlay Grad-CAM embebida como
imagen, no como captura de pantalla de la ventana).

## Alternativas consideradas

- **`tkcap` + `img2pdf`** (descartado): rotos en versiones recientes de Python/Tkinter;
  además, capturar la ventana completa introduce dependencia frágil de resolución de
  pantalla y tema del sistema operativo.
- **`Pillow.ImageGrab` + `reportlab`**: viable pero sigue dependiendo de capturar
  píxeles de pantalla; se descartó a favor de construir el PDF con los datos estructurados
  (`PredictionResult`) en lugar de una imagen de la interfaz.
- **`matplotlib.savefig` a PDF**: añade una dependencia pesada solo para exportar un
  documento, sin beneficio sobre `reportlab`.

## Consecuencias

- `src/detector_neumonia.py.on_export_pdf()` construye el PDF con `reportlab.canvas`,
  incrustando el overlay de Grad-CAM (ya disponible como array `uint8`) y los campos de
  texto (cédula, clase, probabilidad) sin pasar por una captura de pantalla.
- El PDF resultante es reproducible y no depende de la resolución/tema del sistema
  operativo donde corre la GUI.
- `reportlab` queda fijado en el stack técnico del proyecto (sección 1.2 del plan
  maestro) y en `pyproject.toml`.

## Estado

**Aceptado.**