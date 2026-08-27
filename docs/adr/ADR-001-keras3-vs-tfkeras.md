# ADR-001: Keras 3 (Plan A) vs tf-keras / TF_USE_LEGACY_KERAS (Plan B)

## Contexto

`conv_MLP_84.h5` es un modelo legado. `tensorflow==2.20.*` (requisito de Python 3.13 del
proyecto) trae Keras 3 por defecto, cuyo formato interno y API de carga difieren de
Keras 2. Existía riesgo de que `keras.models.load_model()` fallara sobre un `.h5`
entrenado con una versión anterior de Keras, lo que habría obligado a usar el paquete de
compatibilidad `tf-keras` con `TF_USE_LEGACY_KERAS=1` (Plan B).

## Decisión

Usar **Keras 3 nativo (Plan A)**: `tensorflow==2.20.0` + `keras==3.15.1`, sin variables de
entorno de compatibilidad legacy.

## Alternativas consideradas

- **Plan B — `tf-keras` + `TF_USE_LEGACY_KERAS=1`**: habría añadido una dependencia extra
  y un flag de entorno permanente en todo el pipeline de arranque (Dockerfile, CI,
  Makefile), a cambio de ningún beneficio si el Plan A funciona.
- **Reentrenar el modelo en Keras 3 desde cero**: descartado por alcance — el objetivo del
  proyecto es refactorizar arquitectura, no reentrenar el modelo.

## Consecuencias

- Todo el código de `src/load_model.py` y `src/grad_cam.py` usa la API funcional de
  Keras 3 (`keras.Model(model.inputs, [...])`, `tf.GradientTape`) sin capas de
  compatibilidad adicionales.
- Confirmado operativamente en H8: el modelo real carga en 0.6–4.5 s (57 capas) y produce
  inferencias y gradientes de Grad-CAM no nulos con `tensorflow=2.20.0`/`keras=3.15.1`.
- Si en el futuro se necesitara cargar un `.h5` que sí falle en Keras 3, este ADR debe
  actualizarse a "Superado" y reabrirse la evaluación del Plan B.

## Estado

**Aceptado.**