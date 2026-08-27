# Arquitectura — UAO-Neumonia

> ⚠️ Herramienta de apoyo educativo, no es un dispositivo médico certificado.

## 1. Diagrama de secuencia (flujo de una predicción)

```mermaid
sequenceDiagram
    actor U as Usuario
    participant V as Vista (detector_neumonia)
    participant Int as integrator.predict
    participant Pre as preprocess_img
    participant Mod as load_model
    participant GC as grad_cam

    U->>V: Carga imagen y presiona "Predecir"
    V->>V: Lanza inferencia en hilo secundario
    V->>Int: predict(image_rgb)
    Int->>Pre: preprocess(image_rgb)
    Pre-->>Int: batch float32 (1,512,512,1)
    Int->>Mod: load_cnn_model()  [cacheado con lru_cache]
    Mod-->>Int: modelo Keras
    Int->>GC: grad_cam(image_rgb)
    GC->>Mod: load_cnn_model()  [cache hit, sin recarga]
    GC-->>Int: overlay RGB uint8, class_index, probabilidad
    Int-->>V: PredictionResult(label, probability, class_index, heatmap)
    V-->>U: Actualiza UI (clase, probabilidad, mapa de calor)
```

## 2. Diagrama de dependencias entre módulos

```mermaid
graph TD
    detector_neumonia --> integrator
    detector_neumonia --> read_img
    integrator --> preprocess_img
    integrator --> load_model
    integrator --> grad_cam
    grad_cam --> load_model
    preprocess_img --> config
    read_img --> config
    load_model --> config
    grad_cam --> config
    integrator --> config
```

## 3. Justificación del diseño

**`integrator.py` como única puerta de entrada.** La Vista no debe conocer si una
predicción requiere preprocesamiento, carga de modelo o cálculo de Grad-CAM: solo conoce
`predict(image_rgb) -> PredictionResult`. Esto permite que cualquiera de los tres pasos
internos cambie (por ejemplo, sustituir el modelo o el algoritmo de explicabilidad) sin
tocar una sola línea de la GUI, y es lo que hace posible cumplir la regla MVC "la Vista no
importa TensorFlow ni Keras".

**Por qué el modelo se cachea (`@lru_cache(maxsize=1)` en `load_cnn_model`).** Cargar
`conv_MLP_84.h5` toma entre 0.6 s y 4.5 s (medido empíricamente en H8 con el modelo real).
Sin caché, cada clic en "Predecir" repetiría esa carga, degradando la experiencia de uso
interactivo de la GUI. Con caché, la carga ocurre una sola vez por proceso y las
predicciones subsiguientes solo pagan el costo de la inferencia y de Grad-CAM.

**Por qué la Vista corre la inferencia en otro hilo.** Tkinter usa un único hilo de
interfaz (`mainloop`); cualquier operación bloqueante ejecutada en ese hilo congela la
ventana. Como la inferencia + Grad-CAM puede tardar varios segundos, `on_predict()`
delega el trabajo pesado a un hilo secundario y actualiza los widgets desde el hilo
principal al recibir el resultado, evitando que la GUI aparente estar "colgada".

## 4. Regla de dependencias y su verificación automática

Regla congelada (sección 1.1 del plan maestro):

- La Vista (`detector_neumonia.py`) **no** importa `tensorflow`, `keras`, `cv2` ni
  `pydicom`.
- Modelo y Controlador (`load_model.py`, `grad_cam.py`, `read_img.py`,
  `preprocess_img.py`, `integrator.py`) **no** importan `tkinter`.
- `detector_neumonia.py` solo puede hacer `from integrator import predict,
  PredictionResult` y `from read_img import read_image_file`.

`tests/test_architecture.py` verifica esto **estáticamente con el módulo `ast`**: recorre
el árbol de sintaxis de cada archivo de `src/`, extrae los nodos `Import`/`ImportFrom` y
falla el test si aparece una importación prohibida, sin necesidad de ejecutar ni importar
el código (por eso corre incluso sin TensorFlow instalado). Esto convierte la regla de
dependencias en un contrato ejecutable, no solo en una convención documentada.