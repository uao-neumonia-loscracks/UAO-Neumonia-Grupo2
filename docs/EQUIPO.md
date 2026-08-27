| Rol | Persona | Módulos que sube (PRs) | Módulos que revisa | Qué explica en la sustentación |
|---|---|---|---|---|
| **P1 — Tech Lead / DevOps** | Juan (`@JCMelendezT`) | Bootstrap, Docker, CI, Release | PRs de P2 y P3 | Arquitectura MVC, test de arquitectura (AST), Docker/DISPLAY, Gitflow |
| **P2 — Controlador de entrada** | JuanMa (`@juano2024`) | `read_img.py`, `preprocess_img.py`, `docs/PIPELINE.md` | PRs de P1 y P4 | Lectura DICOM (MONOCHROME1, Rescale), pipeline resize→CLAHE→normalize |
| **P3 — Modelo** | Nat (`@nathernandez1189`) | `load_model.py`, `grad_cam.py`, `docs/MODEL_CARD.md` | PRs de P2 y P4 | Carga y caché del `.h5`, algoritmo Grad-CAM, validación del mapeo de clases |
| **P4 — Vista / Integración** | Miguel (`@MiguelDiuza`) | `integrator.py`, `detector_neumonia.py`, `README.md`, `ARCHITECTURE.md` | PRs de P1 y P3 | Orquestación, `PredictionResult`, GUI Tkinter, hilo daemon, PDF/CSV |