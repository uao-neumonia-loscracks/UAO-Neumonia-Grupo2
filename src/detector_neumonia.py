"""Vista Tkinter del detector de neumonía (UAO-Neumonia).

Capa VISTA de la arquitectura MVC estricta del proyecto. Esta capa solo puede
importar del Controlador (`integrator`, `read_img`); tiene prohibido importar
las bibliotecas de aprendizaje profundo, visión artificial o lectura de
archivos médicos, ya que esas dependencias viven exclusivamente en la capa
Modelo (`load_model.py`, `grad_cam.py`).

Advertencia:
    Esta herramienta es de apoyo educativo y NO constituye un dispositivo
    médico certificado ni un diagnóstico clínico definitivo. Toda decisión
    clínica debe ser tomada por personal médico calificado.
"""

from __future__ import annotations

import csv
import logging
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as pdf_canvas

from integrator import PredictionError, PredictionResult, predict
from read_img import read_image_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TAMANO_MINIATURA: tuple[int, int] = (350, 350)
_LONGITUD_MAXIMA_CEDULA: int = 15
_RUTA_HISTORIAL_CSV: Path = Path("historial.csv")
_ENCABEZADO_CSV: tuple[str, str, str, str, str] = (
    "timestamp_iso",
    "cedula",
    "archivo",
    "clase",
    "probabilidad",
)
_EXTENSIONES_SOPORTADAS: tuple[tuple[str, str], ...] = (
    ("Imágenes soportadas", "*.dcm *.dicom *.jpg *.jpeg *.png"),
    ("Todos los archivos", "*.*"),
)
_ADVERTENCIA_LEGAL: str = "Herramienta de apoyo educativo — no constituye diagnóstico médico."
_TITULO_APP: str = "UAO-Neumonía · Detector asistido por IA"
_ANCHO_VENTANA: int = 920
_ALTO_VENTANA: int = 680
_MARGEN_PDF_CM: float = 1.5
_LADO_IMAGEN_PDF_CM: float = 7.0
_DIMENSION_IMAGEN_DUMMY: int = 512
_CANALES_RGB: int = 3


@dataclass
class _EstadoSesion:
    """Contiene el estado mutable de una sesión de uso de la GUI.

    Attributes:
        ruta_archivo: Ruta del archivo de imagen cargado, o None si no hay.
        imagen_original_rgb: Arreglo RGB uint8 de la radiografía cargada.
        imagen_pil: Imagen PIL de la radiografía, usada para mostrarla.
        resultado: Resultado de la última predicción, o None si no se predijo.
    """

    ruta_archivo: Path | None = None
    imagen_original_rgb: np.ndarray | None = None
    imagen_pil: Image.Image | None = None
    resultado: PredictionResult | None = None


class PneumoniaDetectorApp(Tk):
    """Ventana principal de la aplicación de detección de neumonía.

    Orquesta la interacción del usuario con el Controlador (`integrator`,
    `read_img`) sin contener lógica de negocio ni dependencias de
    aprendizaje profundo, visión artificial o lectura DICOM.
    """

    def __init__(self) -> None:
        """Inicializa la ventana, construye los widgets y precarga el modelo."""
        super().__init__()
        self.title(_TITULO_APP)
        self.geometry(f"{_ANCHO_VENTANA}x{_ALTO_VENTANA}")
        self.minsize(_ANCHO_VENTANA, _ALTO_VENTANA)
        self.resizable(width=True, height=True)

        self._estado = _EstadoSesion()
        self._foto_original_tk: ImageTk.PhotoImage | None = None
        self._foto_heatmap_tk: ImageTk.PhotoImage | None = None
        self._botones: list[ttk.Button] = []

        self._construir_widgets()
        self._precargar_modelo_en_background()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _construir_widgets(self) -> None:
        """Crea y organiza todos los widgets de la ventana principal.

        Los paneles de imagen usan un contenedor de tamaño fijo con
        ``pack_propagate(False)`` para que el layout no cambie de altura al
        cargar una radiografía real; de lo contrario, el crecimiento del
        panel de imagen empuja los botones y la barra de estado fuera del
        área visible de la ventana.
        """
        contenedor = ttk.Frame(self, padding=10)
        contenedor.pack(fill="both", expand=True)

        ttk.Label(contenedor, text=_TITULO_APP, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            contenedor,
            text=_ADVERTENCIA_LEGAL,
            font=("Segoe UI", 9, "italic"),
            foreground="firebrick",
        ).pack(anchor="w", pady=(0, 10))

        marco_cedula = ttk.Frame(contenedor)
        marco_cedula.pack(anchor="w", pady=(0, 10))
        ttk.Label(marco_cedula, text="Cédula del paciente:").pack(side="left")
        validador = (self.register(self._validar_cedula), "%P")
        self._entrada_cedula = ttk.Entry(
            marco_cedula, validate="key", validatecommand=validador, width=20
        )
        self._entrada_cedula.pack(side="left", padx=(5, 0))

        marco_imagenes = ttk.Frame(contenedor)
        marco_imagenes.pack(fill="x", pady=(0, 10))

        ancho_panel = _TAMANO_MINIATURA[0] + 20
        alto_panel = _TAMANO_MINIATURA[1] + 20

        marco_izq = ttk.LabelFrame(marco_imagenes, text="Radiografía")
        marco_izq.pack(side="left", padx=(0, 10))
        marco_izq.configure(width=ancho_panel, height=alto_panel)
        marco_izq.pack_propagate(False)
        self._label_imagen_original = ttk.Label(
            marco_izq, text="Sin imagen cargada", anchor="center"
        )
        self._label_imagen_original.pack(fill="both", expand=True)

        marco_der = ttk.LabelFrame(marco_imagenes, text="Mapa de calor (Grad-CAM)")
        marco_der.pack(side="left")
        marco_der.configure(width=ancho_panel, height=alto_panel)
        marco_der.pack_propagate(False)
        self._label_heatmap = ttk.Label(marco_der, text="Sin resultado", anchor="center")
        self._label_heatmap.pack(fill="both", expand=True)

        marco_resultado = ttk.LabelFrame(contenedor, text="Resultado")
        marco_resultado.pack(fill="x", pady=(0, 10))
        self._var_clase = ttk.Label(marco_resultado, text="Clase predicha: —")
        self._var_clase.pack(anchor="w", padx=5, pady=2)
        self._var_probabilidad = ttk.Label(marco_resultado, text="Probabilidad: —")
        self._var_probabilidad.pack(anchor="w", padx=5, pady=2)

        marco_botones = ttk.Frame(contenedor)
        marco_botones.pack(fill="x", pady=(0, 10))
        definiciones_botones = (
            ("Cargar Imagen", self.on_load_image),
            ("Predecir", self.on_predict),
            ("Guardar CSV", self.on_save_csv),
            ("Exportar PDF", self.on_export_pdf),
            ("Borrar", self.on_clear),
        )
        for texto, comando in definiciones_botones:
            boton = ttk.Button(marco_botones, text=texto, command=comando)
            boton.pack(side="left", padx=5)
            self._botones.append(boton)

        self._var_estado = ttk.Label(contenedor, text="Listo.", relief="sunken", anchor="w")
        self._var_estado.pack(fill="x", side="bottom")

    def _validar_cedula(self, texto_propuesto: str) -> bool:
        """Valida en vivo que la cédula contenga solo dígitos y longitud limitada.

        Args:
            texto_propuesto: Contenido que tendría el Entry tras la edición.

        Returns:
            True si el texto propuesto es válido (vacío o dígitos ≤ máximo).
        """
        if texto_propuesto == "":
            return True
        return texto_propuesto.isdigit() and len(texto_propuesto) <= _LONGITUD_MAXIMA_CEDULA

    # ------------------------------------------------------------------
    # Precarga del modelo
    # ------------------------------------------------------------------
    def _precargar_modelo_en_background(self) -> None:
        """Dispara la carga del modelo en un hilo daemon al iniciar la app.

        Se invoca `predict` con una imagen ficticia para forzar la carga y el
        cacheo del modelo (`load_cnn_model` usa `lru_cache`) antes de que el
        usuario realice la primera predicción real, evitando la latencia de
        carga durante la demo en vivo.
        """
        self._fijar_estado("Cargando modelo en segundo plano...")

        def _tarea() -> None:
            """Ejecuta la precarga; no debe tocar widgets desde este hilo."""
            imagen_dummy = np.zeros(
                (_DIMENSION_IMAGEN_DUMMY, _DIMENSION_IMAGEN_DUMMY, _CANALES_RGB),
                dtype=np.uint8,
            )
            try:
                predict(imagen_dummy)
            except (PredictionError, ValueError):
                logger.exception("Fallo al precargar el modelo en segundo plano.")
            except Exception:  # noqa: BLE001 - la precarga nunca debe romper la app
                logger.exception("Error inesperado durante la precarga del modelo.")
            finally:
                self.after(0, lambda: self._fijar_estado("Listo."))

        threading.Thread(target=_tarea, daemon=True).start()

    # ------------------------------------------------------------------
    # Callbacks de botones
    # ------------------------------------------------------------------
    def on_load_image(self) -> None:
        """Abre un diálogo de archivo y carga la imagen seleccionada.

        Delega la lectura al Controlador (`read_img.read_image_file`).
        Cualquier excepción (formato no soportado, archivo corrupto, etc.)
        se captura y se muestra al usuario como un mensaje amigable.
        """
        ruta_str = filedialog.askopenfilename(
            title="Seleccionar radiografía", filetypes=_EXTENSIONES_SOPORTADAS
        )
        if not ruta_str:
            return

        ruta = Path(ruta_str)
        self._fijar_estado(f"Cargando {ruta.name}...")
        try:
            imagen_rgb, imagen_pil = read_image_file(ruta)
        except Exception as exc:  # noqa: BLE001 - traducido a mensaje amigable
            logger.exception("Error al leer el archivo de imagen '%s'.", ruta)
            messagebox.showerror(
                "No se pudo cargar la imagen",
                f"El archivo seleccionado no pudo ser leído.\nDetalle: {exc}",
            )
            self._fijar_estado("Listo.")
            return

        self._estado.ruta_archivo = ruta
        self._estado.imagen_original_rgb = imagen_rgb
        self._estado.imagen_pil = imagen_pil
        self._estado.resultado = None

        self._foto_original_tk = self._crear_miniatura_tk(imagen_pil)
        self._label_imagen_original.configure(image=self._foto_original_tk, text="")
        self._label_heatmap.configure(image="", text="Sin resultado")
        self._foto_heatmap_tk = None
        self._var_clase.configure(text="Clase predicha: —")
        self._var_probabilidad.configure(text="Probabilidad: —")
        self._fijar_estado(f"Imagen cargada: {ruta.name}")

    def on_predict(self) -> None:
        """Ejecuta la inferencia en un hilo separado sin congelar la GUI.

        Tkinter no es thread-safe: sus widgets solo deben modificarse desde el
        hilo principal del event loop. Por eso `predict()` corre en un hilo
        `daemon`, y toda actualización de la interfaz se reprograma con
        `self.after(0, callback)`, que encola la ejecución en el hilo
        principal en la siguiente iteración del loop de eventos.
        """
        if self._estado.imagen_original_rgb is None:
            messagebox.showwarning("Sin imagen", "Debe cargar una imagen antes de predecir.")
            return

        imagen = self._estado.imagen_original_rgb
        self._deshabilitar_botones()
        self._fijar_estado("Procesando...")

        def _tarea() -> None:
            """Corre en un hilo daemon; nunca actualiza widgets directamente."""
            try:
                resultado = predict(imagen)
            except (PredictionError, ValueError) as exc:
                self.after(0, lambda exc=exc: self._on_error_prediccion(exc))
            except Exception as exc:  # noqa: BLE001 - traducido a mensaje amigable
                self.after(0, lambda exc=exc: self._on_error_prediccion(exc))
            else:
                self.after(0, lambda resultado=resultado: self._on_exito_prediccion(resultado))

        threading.Thread(target=_tarea, daemon=True).start()

    def _on_exito_prediccion(self, resultado: PredictionResult) -> None:
        """Actualiza la GUI con un resultado exitoso (llamado desde el hilo principal).

        Args:
            resultado: Resultado devuelto por `integrator.predict`.
        """
        self._estado.resultado = resultado
        imagen_heatmap = Image.fromarray(resultado.heatmap)
        self._foto_heatmap_tk = self._crear_miniatura_tk(imagen_heatmap)
        self._label_heatmap.configure(image=self._foto_heatmap_tk, text="")
        self._var_clase.configure(text=f"Clase predicha: {resultado.label}")
        self._var_probabilidad.configure(text=f"Probabilidad: {resultado.probability:.2f} %")
        self._habilitar_botones()
        self._fijar_estado("Predicción completada.")

    def _on_error_prediccion(self, error: Exception) -> None:
        """Muestra un mensaje amigable ante un error de predicción.

        Args:
            error: Excepción capturada en el hilo de inferencia. El traceback
                completo se envía al log; al usuario solo se le muestra un
                mensaje resumido, sin detalles internos del pipeline.
        """
        logger.exception("Error durante la predicción.", exc_info=error)
        messagebox.showerror(
            "Error de predicción",
            "No fue posible completar la predicción sobre la imagen cargada.\n"
            "Verifique que la imagen sea una radiografía de tórax válida.",
        )
        self._habilitar_botones()
        self._fijar_estado("Listo.")

    def on_save_csv(self) -> None:
        """Anexa el resultado actual a `historial.csv` (crea encabezado si falta)."""
        if self._estado.resultado is None or self._estado.ruta_archivo is None:
            messagebox.showwarning(
                "Sin resultado", "Debe predecir sobre una imagen antes de guardar."
            )
            return

        fila = (
            datetime.now(UTC).isoformat(),
            self._entrada_cedula.get(),
            self._estado.ruta_archivo.name,
            self._estado.resultado.label,
            f"{self._estado.resultado.probability:.2f}",
        )
        try:
            archivo_existe = _RUTA_HISTORIAL_CSV.exists()
            with _RUTA_HISTORIAL_CSV.open(mode="a", newline="", encoding="utf-8") as archivo:
                escritor = csv.writer(archivo)
                if not archivo_existe:
                    escritor.writerow(_ENCABEZADO_CSV)
                escritor.writerow(fila)
        except OSError as exc:
            logger.exception("Error al escribir en el historial CSV.")
            messagebox.showerror(
                "Error al guardar", f"No se pudo escribir en el archivo CSV.\n{exc}"
            )
            return

        self._fijar_estado(f"Historial actualizado en {_RUTA_HISTORIAL_CSV}.")
        messagebox.showinfo("Guardado", "El registro se guardó correctamente.")

    def on_export_pdf(self) -> None:
        """Genera un reporte en PDF con reportlab con ambas imágenes y los datos."""
        if (
            self._estado.resultado is None
            or self._estado.imagen_pil is None
            or self._estado.ruta_archivo is None
        ):
            messagebox.showwarning(
                "Sin resultado", "Debe predecir sobre una imagen antes de exportar."
            )
            return

        ruta_destino_str = filedialog.asksaveasfilename(
            title="Guardar reporte PDF",
            defaultextension=".pdf",
            filetypes=(("Documento PDF", "*.pdf"),),
        )
        if not ruta_destino_str:
            return

        try:
            self._generar_reporte_pdf(Path(ruta_destino_str))
        except Exception as exc:  # noqa: BLE001 - traducido a mensaje amigable
            logger.exception("Error al generar el reporte PDF.")
            messagebox.showerror("Error al exportar", f"No se pudo generar el PDF.\n{exc}")
            return

        self._fijar_estado(f"Reporte exportado a {ruta_destino_str}.")
        messagebox.showinfo("Exportado", "El reporte PDF se generó correctamente.")

    def _generar_reporte_pdf(self, ruta_destino: Path) -> None:
        """Construye el documento PDF del reporte clínico educativo.

        Args:
            ruta_destino: Ruta de archivo donde se guardará el PDF.

        Raises:
            RuntimeError: Si no hay resultado ni imagen cargados (no debería
                ocurrir, ya que `on_export_pdf` valida antes de llamar aquí).
        """
        resultado = self._estado.resultado
        imagen_pil = self._estado.imagen_pil
        ruta_archivo = self._estado.ruta_archivo
        if resultado is None or imagen_pil is None or ruta_archivo is None:
            raise RuntimeError("Falta información de sesión para generar el PDF.")

        imagen_heatmap = Image.fromarray(resultado.heatmap)

        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_original,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_heatmap,
        ):
            ruta_tmp_original = Path(tmp_original.name)
            ruta_tmp_heatmap = Path(tmp_heatmap.name)

        imagen_pil.convert("RGB").save(ruta_tmp_original)
        imagen_heatmap.convert("RGB").save(ruta_tmp_heatmap)

        try:
            lienzo = pdf_canvas.Canvas(str(ruta_destino), pagesize=letter)
            ancho_pagina, alto_pagina = letter
            margen = _MARGEN_PDF_CM * cm
            y_cursor = alto_pagina - margen

            lienzo.setFont("Helvetica-Bold", 16)
            lienzo.drawString(margen, y_cursor, "UAO-Neumonía · Reporte de apoyo diagnóstico")
            y_cursor -= 0.8 * cm

            lienzo.setFont("Helvetica", 10)
            lienzo.drawString(
                margen, y_cursor, f"Cédula del paciente: {self._entrada_cedula.get() or '—'}"
            )
            y_cursor -= 0.5 * cm
            lienzo.drawString(
                margen,
                y_cursor,
                f"Fecha y hora: {datetime.now(UTC).isoformat()}",
            )
            y_cursor -= 0.5 * cm
            lienzo.drawString(margen, y_cursor, f"Archivo analizado: {ruta_archivo.name}")
            y_cursor -= 1.0 * cm

            lado = _LADO_IMAGEN_PDF_CM * cm
            lienzo.drawImage(
                str(ruta_tmp_original), margen, y_cursor - lado, width=lado, height=lado
            )
            lienzo.drawImage(
                str(ruta_tmp_heatmap),
                margen + lado + 0.5 * cm,
                y_cursor - lado,
                width=lado,
                height=lado,
            )
            y_cursor -= lado + 0.8 * cm

            lienzo.setFont("Helvetica-Bold", 12)
            lienzo.drawString(margen, y_cursor, f"Clase predicha: {resultado.label}")
            y_cursor -= 0.6 * cm
            lienzo.drawString(margen, y_cursor, f"Probabilidad: {resultado.probability:.2f} %")

            lienzo.setFont("Helvetica-Oblique", 8)
            lienzo.drawString(margen, margen, _ADVERTENCIA_LEGAL)

            lienzo.save()
        finally:
            ruta_tmp_original.unlink(missing_ok=True)
            ruta_tmp_heatmap.unlink(missing_ok=True)

    def on_clear(self) -> None:
        """Limpia campos, imágenes y estado de sesión, previa confirmación del usuario."""
        if not messagebox.askokcancel(
            "Confirmar borrado", "¿Desea borrar la imagen y los resultados actuales?"
        ):
            return

        self._estado = _EstadoSesion()
        self._foto_original_tk = None
        self._foto_heatmap_tk = None
        self._entrada_cedula.delete(0, "end")
        self._label_imagen_original.configure(image="", text="Sin imagen cargada")
        self._label_heatmap.configure(image="", text="Sin resultado")
        self._var_clase.configure(text="Clase predicha: —")
        self._var_probabilidad.configure(text="Probabilidad: —")
        self._fijar_estado("Listo.")

    # ------------------------------------------------------------------
    # Utilidades privadas de UI
    # ------------------------------------------------------------------
    def _crear_miniatura_tk(self, imagen: Image.Image) -> ImageTk.PhotoImage:
        """Genera una miniatura de la imagen apta para mostrarse en la GUI.

        Args:
            imagen: Imagen PIL de origen (radiografía u overlay de Grad-CAM).

        Returns:
            Objeto `ImageTk.PhotoImage` listo para asignarse a un `Label`.
        """
        copia = imagen.copy()
        copia.thumbnail(_TAMANO_MINIATURA, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(copia)

    def _deshabilitar_botones(self) -> None:
        """Deshabilita todos los botones de acción mientras se procesa una tarea."""
        for boton in self._botones:
            boton.configure(state="disabled")

    def _habilitar_botones(self) -> None:
        """Reactiva todos los botones de acción tras finalizar una tarea."""
        for boton in self._botones:
            boton.configure(state="normal")

    def _fijar_estado(self, mensaje: str) -> None:
        """Actualiza el texto de la barra de estado inferior.

        Args:
            mensaje: Texto a mostrar (por ejemplo, "listo", "procesando...").
        """
        self._var_estado.configure(text=mensaje)


def main() -> None:
    """Punto de entrada de la aplicación de escritorio."""
    app = PneumoniaDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
