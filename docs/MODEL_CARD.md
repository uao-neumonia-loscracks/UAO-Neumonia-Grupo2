# Model Card — conv_MLP_84.h5 (UAO-Neumonia)

> ⚠️ **Advertencia médica**: este modelo es de uso educativo exclusivamente. No está
> certificado como dispositivo médico, no ha sido validado clínicamente y no debe usarse
> para decisiones de diagnóstico o tratamiento reales.

## 1. Detalles del modelo

| Campo | Valor | Fuente |
|---|---|---|
| Nombre de archivo | `conv_MLP_84.h5` | Repositorio (excluido de git) |
| Tipo de arquitectura | CNN basada en Pasa et al. (bloques convolucionales con conexión "skip") | README original del proyecto base [2] |
| Bloques convolucionales | 5 bloques, cada uno con 2 convoluciones secuenciales 3×3 + 1 conexión "skip" 3×3 | README original [2] |
| Filtros por bloque | 16, 32, 48, 64, 80 | README original [2] |
| Pooling | MaxPooling tras cada bloque; AveragePooling tras el último bloque | README original [2] |
| Capas densas finales | 1024 → 1024 → 3 (softmax) | README original [2] |
| Regularización | 3 capas Dropout (20%): en bloques 4 y 5, y tras la 1ª capa densa | README original [2] |
| Número total de capas | **57** | Verificado empíricamente en H8 (log de `smoke_test.py` con modelo real: "Modelo cargado en 0.6–4.5s (57 capas)") |
| Número de parámetros entrenables | **No disponible** | No se registró en este hilo. Se obtiene con `model.count_params()` — expuesto por `load_model.model_summary_text(model)`. Debe correrse y pegarse aquí. |
| Nombre exacto de la última capa Conv2D (objetivo de Grad-CAM) | **No disponible** | Debe extraerse de `docs/handoffs/HANDOFF-H0.md` o recalcularse con `load_model.get_last_conv_layer_name(model)`. |
| SHA256 del archivo `.h5` | **No disponible** | No se dispone del handoff H0 en este hilo. Calcular con `uv run python -c "from load_model import compute_sha256; print(compute_sha256('models/conv_MLP_84.h5'))"` y actualizar esta tabla y `models/README.md`. |

## 2. Contrato de entrada/salida

| Etapa | Shape | dtype | Rango | Estado |
|---|---|---|---|---|
| Entrada al modelo (`preprocess`) | `(1, 512, 512, 1)` | `float32` | `[0.0, 1.0]` | Contrato congelado (plan maestro §1.3/5.4); confirmado funcionalmente en H8 mediante inferencias reales exitosas. |
| Salida del modelo | vector de 3 probabilidades (softmax) | `float32` | suman ≈ 1.0 | Consistente con `CLASS_LABELS` de 3 elementos; el valor literal de `model.output_shape` **no** fue confirmado explícitamente en este hilo — pendiente de `HANDOFF-H0.md`. |
| Mapa Grad-CAM | `(512, 512)` | `float32` | `[0.0, 1.0]` | Contrato congelado, verificado en H8 (Grad-CAM calculado sobre inferencias reales). |
| Overlay Grad-CAM | `(512, 512, 3)` | `uint8` | `[0, 255]` | Contrato congelado, verificado en H8. |

## 3. Uso previsto

Apoyo educativo a la interpretación de radiografías de tórax en un contexto de
enseñanza de deep learning e interpretabilidad (Grad-CAM). **No** apto para uso clínico,
triage real ni como única fuente de decisión médica.

## 4. Mapeo de clases — ⚠️ NO CONGELADO, EN INVESTIGACIÓN

El código legado asume `0 → bacteriana`, `1 → normal`, `2 → viral`. El plan maestro exige
verificar este mapeo empíricamente antes de documentarlo como definitivo (sección 1.4).

**Hallazgo urgente heredado de H8** (sin resolver a la fecha de este Model Card):

> La imagen `person1710_bacteria_4526.jpeg` — cuyo nombre de archivo, según la convención
> del dataset de origen, indica clase esperada "bacteriana" — fue clasificada por el
> modelo como **viral con 85.92% de probabilidad**.

Esto es evidencia real (no hipotética) de que **al menos uno** de estos escenarios es
cierto:

1. El mapeo `0/1/2` heredado del código original es incorrecto para este `.h5` específico.
2. El modelo tiene baja exactitud real en esta clase, independientemente del mapeo.
3. La imagen de prueba está mal etiquetada en el dataset de origen (menos probable, pero
   no descartable sin auditoría).

**Estado**: el mapeo de clases de este Model Card se marca como **NO VALIDADO**. No debe
usarse como fuente de verdad hasta que se ejecute lo indicado en la sección 6.

## 5. Datos de entrenamiento y evaluación

| Campo | Valor |
|---|---|
| Dataset de entrenamiento | **No disponible** en este hilo — el proyecto reutiliza un modelo ya entrenado (`conv_MLP_84.h5`); no se tiene registro del dataset ni del split usados por los autores originales. |
| Métricas de evaluación (exactitud, precisión, recall, F1 por clase) | **No disponible** — no se han calculado en ningún hilo hasta la fecha. Ver plan de obtención en la sección 6. |
| Matriz de confusión | **No disponible** — pendiente, ver sección 6. |

> Nota de integridad: se optó deliberadamente por **no inventar** estos números. Un
> Model Card con métricas ficticias sería más peligroso que uno que declara
> explícitamente sus vacíos, dado el contexto clínico del proyecto.

## 6. Cómo obtener los datos faltantes (plan de acción)

1. Reunir un lote de ≥30 imágenes por clase de fuente conocida (p. ej. el dataset original
   de Kermany et al. usado por el proyecto base, cuya convención de nombres
   `personXXXX_bacteria_YYYY.jpeg` / `personXXXX_virus_YYYY.jpeg` / `personXXXX_NORMAL_...`
   permite derivar la etiqueta real).
2. Ejecutar el pipeline completo (`read_img` → `preprocess` → modelo) sobre cada imagen y
   registrar `(etiqueta_esperada, class_index_predicho, probabilidad)`.
3. Construir una matriz de confusión 3×3 y calcular exactitud, precisión y recall por
   clase.
4. Si la matriz de confusión muestra que la diagonal real no corresponde a
   `{0:bacteriana, 1:normal, 2:viral}` sino a una permutación distinta, corregir
   `CLASS_LABELS` en `src/config.py` y registrar un ADR documentando el cambio.
5. Solo entonces, actualizar este Model Card reemplazando "NO VALIDADO" por el mapeo
   confirmado y las métricas reales.

## 7. Consideraciones éticas y limitaciones

- Riesgo clínico principal: un mapeo de clases incorrecto produce un diagnóstico
  incorrecto con apariencia de confianza alta (ver hallazgo de la sección 4).
- El modelo no fue evaluado para sesgos demográficos (edad, sexo, equipo de adquisición
  DICOM) — no disponible, no evaluado en ningún hilo.
- Grad-CAM explica *dónde* mira el modelo, no *por qué* es correcto; una región anatómica
  plausible resaltada no garantiza una predicción correcta.