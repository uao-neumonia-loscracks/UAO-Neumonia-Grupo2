"""Tests de arquitectura MVC mediante análisis estático con ``ast``.

Verifica, sin ejecutar el código, que se respetan las fronteras de capas
definidas en el plan maestro: la Vista no importa TensorFlow/OpenCV/pydicom,
el Modelo y el Controlador no importan tkinter, y que todos los módulos y
funciones públicas de ``src/`` tienen docstring y anotaciones de retorno.
"""

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

FORBIDDEN_IN_VIEW = {"tensorflow", "keras", "tf_keras", "cv2", "pydicom"}
MODEL_CONTROLLER_MODULES = [
    "read_img.py",
    "preprocess_img.py",
    "load_model.py",
    "grad_cam.py",
    "integrator.py",
]


def _imported_module_names(tree: ast.Module) -> set[str]:
    """Extrae los nombres raíz de todos los módulos importados en un AST.

    Args:
        tree: Árbol de sintaxis abstracta del módulo analizado.

    Returns:
        set[str]: Conjunto de nombres raíz de paquetes importados
        (por ejemplo, ``"tensorflow"`` para ``import tensorflow as tf``).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _parse_module(path: Path) -> ast.Module:
    """Parsea un archivo fuente Python a su árbol de sintaxis abstracta.

    Args:
        path: Ruta al archivo ``.py`` a parsear.

    Returns:
        ast.Module: Árbol de sintaxis abstracta del archivo.
    """
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_vista_no_importa_tensorflow_ni_dependencias_de_modelo() -> None:
    """La Vista (detector_neumonia.py) no debe importar TF/Keras/cv2/pydicom."""
    tree = _parse_module(SRC_DIR / "detector_neumonia.py")
    imports = _imported_module_names(tree)
    interseccion = imports & FORBIDDEN_IN_VIEW
    assert not interseccion, f"La Vista importa dependencias prohibidas: {interseccion}"


@pytest.mark.parametrize("filename", MODEL_CONTROLLER_MODULES)
def test_modelo_y_controlador_no_importan_tkinter(filename: str) -> None:
    """Los módulos de Modelo y Controlador no deben importar tkinter."""
    tree = _parse_module(SRC_DIR / filename)
    imports = _imported_module_names(tree)
    assert "tkinter" not in imports, f"{filename} importa tkinter (prohibido)"


def test_integrator_no_importa_tkinter() -> None:
    """integrator.py, como orquestador del Controlador, no debe importar tkinter."""
    tree = _parse_module(SRC_DIR / "integrator.py")
    imports = _imported_module_names(tree)
    assert "tkinter" not in imports


def test_todos_los_modulos_tienen_docstring_de_modulo() -> None:
    """Cada módulo en src/ debe tener un docstring de módulo no vacío."""
    sin_docstring = []
    for path in sorted(SRC_DIR.glob("*.py")):
        tree = _parse_module(path)
        if not ast.get_docstring(tree):
            sin_docstring.append(path.name)
    assert not sin_docstring, f"Módulos sin docstring: {sin_docstring}"


def test_funciones_publicas_tienen_docstring_y_anotacion_de_retorno() -> None:
    """Toda función/método público en src/ debe tener docstring y anotación de retorno."""
    violaciones = []
    for path in sorted(SRC_DIR.glob("*.py")):
        tree = _parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    violaciones.append(f"{path.name}::{node.name} sin docstring")
                if node.returns is None:
                    violaciones.append(f"{path.name}::{node.name} sin anotación de retorno")
    assert not violaciones, f"Violaciones encontradas: {violaciones}"
