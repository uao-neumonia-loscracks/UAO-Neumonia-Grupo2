# Modelo entrenado — `conv_MLP_84.h5`

> ⚠️ **Advertencia**: esta herramienta es de apoyo educativo, **NO es un dispositivo médico certificado**.

## Cómo obtenerlo

El archivo `conv_MLP_84.h5` **no está en este repositorio** (excluido por `.gitignore` por su tamaño y por política del proyecto). Debe obtenerse por el canal privado acordado por el equipo (Drive/entrega docente) y ubicarse localmente.

## Dónde ubicarlo

Coloca el archivo en:




## Verificación de integridad (SHA-256)

> d9548ff6fd2c29cdd15328ff95e1ad1fc8c7226e04e86c5890526bea2215e18e El SHA-256 real de `conv_MLP_84.h5` todavía
> no se ha registrado en ningún handoff ni ADR del proyecto (ver TODO abierto
> en `HANDOFF-H3.md`, sección 1). **No se debe inventar este valor.** Cuando
> el equipo ejecute `spike_compat.py` / `spike_compat_legacy.py` en H0 (o lo
> reejecute), reemplace la línea `SHA256_ESPERADO` de abajo por el valor real
> y elimine esta advertencia.

Calcula el hash del archivo que descargaste y compáralo contra el valor
esperado antes de usarlo:

```bash
uv run python -c "import load_model as lm; print(lm.compute_sha256('models/conv_MLP_84.h5'))"
```

```text
SHA256_ESPERADO = "d9548ff6fd2c29cdd15328ff95e1ad1fc8c7226e04e86c5890526bea2215e18e"
```

También puedes verificar programáticamente con `verify_model_integrity`:

```bash
uv run python -c "
import load_model as lm
ok = lm.verify_model_integrity('models/conv_MLP_84.h5', expected_sha256='d9548ff6fd2c29cdd15328ff95e1ad1fc8c7226e04e86c5890526bea2215e18e')
print('Integridad OK' if ok else 'Integridad FALLIDA')"
```

Si el hash no coincide, **no uses el archivo**: puede estar corrupto, truncado
o ser una versión distinta del modelo entrenado, y las predicciones (y el
mapeo de clases verificado en H0) dejarían de ser confiables.




