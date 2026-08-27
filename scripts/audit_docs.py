"""Audita docstrings y type hints en el código fuente de ``src/``.

Recorre estáticamente (usando el módulo ``ast``, sin importar el código) todos los
módulos de ``src/`` del proyecto UAO-Neumonia y reporta funciones, clases y módulos
públicos que no cumplan los requisitos de calidad del proyecto: docstring obligatorio
y anotaciones de tipo completas en firmas públicas.

Uso:
    uv run python scripts/audit_docs.py

Código de salida:
    0 si el 100% de los elementos auditados cumple los requisitos.
    1 si se encuentra al menos un incumplimiento.
"""

from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@dataclass(frozen=True)
class Finding:
    """Representa un hallazgo de incumplimiento de documentación o tipado.

    Attributes:
        path: Ruta del archivo donde se encontró el problema.
        line: Número de línea del nodo AST afectado.
        kind: Tipo de elemento ("módulo", "clase" o "función").
        name: Nombre del elemento afectado.
        reason: Motivo del hallazgo (docstring o anotaciones faltantes).
    """

    path: Path
    line: int
    kind: str
    name: str
    reason: str


def _is_public(name: str) -> bool:
    """Determina si un nombre corresponde a un elemento público auditable.

    Args:
        name: Nombre del elemento (función, clase o método).

    Returns:
        ``True`` si el nombre no inicia con guion bajo (o es ``__init__``),
        ``False`` en caso contrario.
    """
    return not name.startswith("_") or name == "__init__"


def _check_function(node: ast.FunctionDef | ast.AsyncFunctionDef, path: Path) -> list[Finding]:
    """Audita una función o método en busca de docstring y type hints.

    Args:
        node: Nodo AST de la función o método a auditar.
        path: Ruta del archivo que contiene el nodo.

    Returns:
        Lista de hallazgos encontrados para esta función (vacía si cumple).
    """
    findings: list[Finding] = []
    if not _is_public(node.name):
        return findings

    if ast.get_docstring(node) is None:
        findings.append(Finding(path, node.lineno, "función", node.name, "sin docstring"))

    if node.returns is None and node.name != "__init__":
        findings.append(
            Finding(path, node.lineno, "función", node.name, "sin anotación de retorno")
        )

    for arg in (*node.args.args, *node.args.kwonlyargs):
        if arg.arg in {"self", "cls"}:
            continue
        if arg.annotation is None:
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    "función",
                    node.name,
                    f"parámetro '{arg.arg}' sin type hint",
                )
            )
    return findings


def _check_class(node: ast.ClassDef, path: Path) -> list[Finding]:
    """Audita una clase y sus métodos públicos.

    Args:
        node: Nodo AST de la clase a auditar.
        path: Ruta del archivo que contiene el nodo.

    Returns:
        Lista de hallazgos encontrados para la clase y sus métodos.
    """
    findings: list[Finding] = []
    if ast.get_docstring(node) is None:
        findings.append(Finding(path, node.lineno, "clase", node.name, "sin docstring"))

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings.extend(_check_function(item, path))
    return findings


def audit_module(path: Path) -> list[Finding]:
    """Audita un único módulo fuente en busca de incumplimientos.

    Args:
        path: Ruta al archivo ``.py`` a auditar.

    Returns:
        Lista de hallazgos encontrados en el módulo.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[Finding] = []

    if ast.get_docstring(tree) is None:
        findings.append(Finding(path, 1, "módulo", path.name, "sin docstring de módulo"))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings.extend(_check_function(node, path))
        elif isinstance(node, ast.ClassDef):
            findings.extend(_check_class(node, path))

    return findings


def audit_src(src_dir: Path) -> list[Finding]:
    """Audita recursivamente todos los módulos ``.py`` de un directorio.

    Args:
        src_dir: Directorio raíz del código fuente a auditar.

    Returns:
        Lista consolidada de hallazgos de todos los módulos encontrados.
    """
    findings: list[Finding] = []
    for py_file in sorted(src_dir.rglob("*.py")):
        findings.extend(audit_module(py_file))
    return findings


def main() -> None:
    """Punto de entrada del script de auditoría.

    Recorre ``src/``, reporta los hallazgos por consola y termina el proceso con
    código de salida ``1`` si hay al menos un incumplimiento, o ``0`` si el 100% de
    los elementos públicos está documentado y tipado.
    """
    if not SRC_DIR.exists():
        logger.error("No existe el directorio %s", SRC_DIR)
        sys.exit(1)

    findings = audit_src(SRC_DIR)

    if not findings:
        print("OK: 100% de funciones documentadas")
        sys.exit(0)

    logger.warning("Se encontraron %d incumplimientos:", len(findings))
    for finding in findings:
        rel = finding.path.relative_to(SRC_DIR.parent)
        print(f"  {rel}:{finding.line} [{finding.kind}] {finding.name} -> {finding.reason}")

    sys.exit(1)


if __name__ == "__main__":
    main()
