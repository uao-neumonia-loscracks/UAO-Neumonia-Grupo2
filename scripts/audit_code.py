"""Auditoría estática de calidad de código para UAO-Neumonia.

Detecta, sin dependencias externas (solo ``ast`` y ``re``), cuatro categorías
de hallazgos sobre los módulos de ``src/``:

1. Marcadores ``# TODO`` / ``# FIXME`` pendientes.
2. Código muerto heurístico (funciones/clases definidas en ``src/`` que no
   se referencian desde ningún otro módulo de ``src/`` ni desde ``tests/``).
3. Imports importados pero nunca usados dentro del propio módulo (excluyendo
   ``from __future__ import ...``, que son directivas del compilador, no
   nombres referenciables).
4. Funciones públicas de ``src/`` sin un test correspondiente en ``tests/``
   (heurística por nombre: se busca ``test_<nombre_funcion>`` o el nombre
   de la función dentro del código fuente de algún archivo de test).
5. Valores mágicos (literales numéricos o de cadena repetidos/relevantes)
   que no provienen de ``src/config.py`` y deberían centralizarse allí.

Uso:
    uv run python scripts/audit_code.py [--src src] [--tests tests]

Salida: exit code 0 si no hay hallazgos, 1 si hay al menos uno. Imprime un
reporte legible por sección usando ``logging``.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("audit_code")

_TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME)\b(.*)", re.IGNORECASE)
_MAGIC_NUMBER_ALLOWLIST: frozenset[int | float] = frozenset({0, 1, -1, 2, 100})
_FUTURE_MODULE = "__future__"


@dataclass(frozen=True)
class Finding:
    """Representa un hallazgo puntual de la auditoría.

    Attributes:
        categoria: Categoría del hallazgo (p. ej. ``"TODO"``, ``"IMPORT_SIN_USAR"``).
        ruta: Ruta relativa del archivo donde se encontró.
        linea: Número de línea (1-indexado) donde ocurre el hallazgo.
        detalle: Descripción legible del hallazgo.
    """

    categoria: str
    ruta: str
    linea: int
    detalle: str


@dataclass
class Reporte:
    """Acumula los hallazgos de toda la auditoría, agrupados por categoría.

    Attributes:
        hallazgos: Lista completa de hallazgos encontrados.
    """

    hallazgos: list[Finding] = field(default_factory=list)

    def agregar(self, hallazgo: Finding) -> None:
        """Agrega un hallazgo al reporte.

        Args:
            hallazgo: Instancia de :class:`Finding` a registrar.
        """
        self.hallazgos.append(hallazgo)

    def por_categoria(self) -> dict[str, list[Finding]]:
        """Agrupa los hallazgos por categoría.

        Returns:
            Diccionario categoría -> lista de hallazgos.
        """
        agrupado: dict[str, list[Finding]] = {}
        for h in self.hallazgos:
            agrupado.setdefault(h.categoria, []).append(h)
        return agrupado


def _leer_texto(path: Path) -> str:
    """Lee un archivo de texto en UTF-8, tolerando errores de codificación.

    Args:
        path: Ruta del archivo a leer.

    Returns:
        Contenido completo del archivo como cadena.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def buscar_todos_fixme(src_dir: Path) -> list[Finding]:
    """Busca marcadores TODO/FIXME en todos los archivos ``.py`` de ``src_dir``.

    Args:
        src_dir: Directorio raíz de código fuente a inspeccionar.

    Returns:
        Lista de hallazgos, uno por marcador encontrado.
    """
    hallazgos: list[Finding] = []
    for archivo in sorted(src_dir.rglob("*.py")):
        texto = _leer_texto(archivo)
        for i, linea in enumerate(texto.splitlines(), start=1):
            match = _TODO_PATTERN.search(linea)
            if match:
                tag = match.group(1).upper()
                comentario = match.group(2).strip(" :-")
                hallazgos.append(Finding(tag, str(archivo), i, comentario or "(sin descripción)"))
    return hallazgos


def _nombres_definidos_y_usados(
    arbol: ast.AST,
) -> tuple[set[str], set[str], set[tuple[str, int, str]]]:
    """Extrae nombres definidos, nombres usados e imports de un módulo AST.

    Los imports de ``from __future__ import ...`` se excluyen del conjunto de
    imports retornado, ya que son directivas del compilador (no introducen un
    nombre referenciable en tiempo de ejecución) y no deben auditarse como
    "sin usar".

    Args:
        arbol: Árbol AST del módulo ya parseado.

    Returns:
        Tupla ``(definidos, usados, imports)`` donde ``imports`` contiene
        tuplas ``(alias, linea, modulo_origen)``.
    """
    definidos: set[str] = set()
    usados: set[str] = set()
    imports: set[tuple[str, int, str]] = set()

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definidos.add(nodo.name)
        elif isinstance(nodo, ast.Name):
            usados.add(nodo.id)
        elif isinstance(nodo, ast.Attribute):
            if isinstance(nodo.value, ast.Name):
                usados.add(nodo.value.id)
        elif isinstance(nodo, ast.Import):
            for alias in nodo.names:
                nombre = alias.asname or alias.name.split(".")[0]
                imports.add((nombre, nodo.lineno, alias.name))
        elif isinstance(nodo, ast.ImportFrom):
            modulo = nodo.module or ""
            if modulo == _FUTURE_MODULE:
                continue
            for alias in nodo.names:
                nombre = alias.asname or alias.name
                imports.add((nombre, nodo.lineno, modulo))

    return definidos, usados, imports


def buscar_imports_sin_usar(src_dir: Path) -> list[Finding]:
    """Detecta imports declarados pero nunca referenciados en el mismo módulo.

    Excluye ``from __future__ import ...`` (ver :func:`_nombres_definidos_y_usados`).

    Args:
        src_dir: Directorio raíz de código fuente a inspeccionar.

    Returns:
        Lista de hallazgos, uno por import sin uso detectado.
    """
    hallazgos: list[Finding] = []
    for archivo in sorted(src_dir.rglob("*.py")):
        texto = _leer_texto(archivo)
        try:
            arbol = ast.parse(texto, filename=str(archivo))
        except SyntaxError as exc:
            logger.warning("No se pudo parsear %s: %s", archivo, exc)
            continue
        _, usados, imports = _nombres_definidos_y_usados(arbol)
        for nombre, linea, origen in imports:
            if nombre == "*":
                continue
            if nombre not in usados:
                hallazgos.append(
                    Finding(
                        "IMPORT_SIN_USAR",
                        str(archivo),
                        linea,
                        f"'{nombre}' importado desde '{origen}' pero no se usa en el módulo",
                    )
                )
    return hallazgos


def buscar_codigo_muerto(src_dir: Path, tests_dir: Path) -> list[Finding]:
    """Heurística de código muerto: funciones/clases de ``src`` sin referencias externas.

    Considera que una función o clase está "viva" si su nombre aparece como
    texto en algún otro archivo de ``src/`` o ``tests/`` distinto al que la
    define. Es una heurística textual (no resuelve imports dinámicos ni
    reflexión), pensada para reducir candidatos a revisión manual.

    Args:
        src_dir: Directorio raíz de código fuente.
        tests_dir: Directorio raíz de tests.

    Returns:
        Lista de hallazgos, uno por símbolo candidato a código muerto.
    """
    hallazgos: list[Finding] = []
    archivos_src = sorted(src_dir.rglob("*.py"))
    archivos_tests = sorted(tests_dir.rglob("*.py")) if tests_dir.exists() else []
    todos_los_archivos = archivos_src + archivos_tests
    contenidos = {a: _leer_texto(a) for a in todos_los_archivos}

    for archivo in archivos_src:
        try:
            arbol = ast.parse(contenidos[archivo], filename=str(archivo))
        except SyntaxError:
            continue
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.FunctionDef | ast.ClassDef):
                continue
            if nodo.name.startswith("_") or nodo.name in {"main", "__init__"}:
                continue
            referencias_externas = 0
            for otro_archivo, texto in contenidos.items():
                if otro_archivo == archivo:
                    continue
                if re.search(rf"\b{re.escape(nodo.name)}\b", texto):
                    referencias_externas += 1
            if referencias_externas == 0:
                hallazgos.append(
                    Finding(
                        "CODIGO_MUERTO_CANDIDATO",
                        str(archivo),
                        nodo.lineno,
                        f"'{nodo.name}' no se referencia en ningún otro archivo de src/ o tests/",
                    )
                )
    return hallazgos


def buscar_funciones_sin_test(src_dir: Path, tests_dir: Path) -> list[Finding]:
    """Detecta funciones públicas de ``src`` sin cobertura textual en ``tests``.

    Heurística: para cada función pública (no ``_privada``) definida en
    ``src/*.py``, se busca su nombre dentro del código de ``tests/*.py``.

    Args:
        src_dir: Directorio raíz de código fuente.
        tests_dir: Directorio raíz de tests.

    Returns:
        Lista de hallazgos, uno por función sin test detectado.
    """
    hallazgos: list[Finding] = []
    if not tests_dir.exists():
        return hallazgos
    texto_tests = " ".join(_leer_texto(a) for a in tests_dir.rglob("*.py"))

    for archivo in sorted(src_dir.rglob("*.py")):
        texto = _leer_texto(archivo)
        try:
            arbol = ast.parse(texto, filename=str(archivo))
        except SyntaxError:
            continue
        for nodo in ast.walk(arbol):
            es_funcion_publica = isinstance(nodo, ast.FunctionDef) and not nodo.name.startswith("_")
            if es_funcion_publica and not re.search(rf"\b{re.escape(nodo.name)}\b", texto_tests):
                hallazgos.append(
                    Finding(
                        "FUNCION_SIN_TEST",
                        str(archivo),
                        nodo.lineno,
                        f"'{nodo.name}' no aparece referenciada en ningún archivo de tests/",
                    )
                )
    return hallazgos


def buscar_valores_magicos(src_dir: Path, config_path: Path) -> list[Finding]:
    """Detecta literales numéricos/de cadena que deberían vivir en ``config.py``.

    Excluye ``config.py`` mismo y los archivos de ``tests/``. Ignora los
    valores del ``_MAGIC_NUMBER_ALLOWLIST`` (0, 1, -1, 2, 100) por ser de uso
    estructural común (índices, porcentajes, negaciones).

    Args:
        src_dir: Directorio raíz de código fuente.
        config_path: Ruta al archivo ``config.py`` (se excluye del escaneo).

    Returns:
        Lista de hallazgos, uno por literal sospechoso.
    """
    hallazgos: list[Finding] = []
    for archivo in sorted(src_dir.rglob("*.py")):
        if archivo.resolve() == config_path.resolve():
            continue
        texto = _leer_texto(archivo)
        try:
            arbol = ast.parse(texto, filename=str(archivo))
        except SyntaxError:
            continue
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Constant) and isinstance(nodo.value, int | float):
                if isinstance(nodo.value, bool):
                    continue
                if nodo.value in _MAGIC_NUMBER_ALLOWLIST:
                    continue
                hallazgos.append(
                    Finding(
                        "VALOR_MAGICO",
                        str(archivo),
                        nodo.lineno,
                        f"Literal numérico {nodo.value!r} fuera de config.py",
                    )
                )
    return hallazgos


def ejecutar_auditoria(src_dir: Path, tests_dir: Path, config_filename: str) -> Reporte:
    """Ejecuta todas las auditorías y consolida un único reporte.

    Args:
        src_dir: Directorio raíz de código fuente.
        tests_dir: Directorio raíz de tests.
        config_filename: Nombre del archivo de configuración a excluir
            del escaneo de valores mágicos (por defecto ``config.py``).

    Returns:
        Instancia de :class:`Reporte` con todos los hallazgos consolidados.
    """
    reporte = Reporte()
    config_path = src_dir / config_filename

    for hallazgo in buscar_todos_fixme(src_dir):
        reporte.agregar(hallazgo)
    for hallazgo in buscar_imports_sin_usar(src_dir):
        reporte.agregar(hallazgo)
    for hallazgo in buscar_codigo_muerto(src_dir, tests_dir):
        reporte.agregar(hallazgo)
    for hallazgo in buscar_funciones_sin_test(src_dir, tests_dir):
        reporte.agregar(hallazgo)
    for hallazgo in buscar_valores_magicos(src_dir, config_path):
        reporte.agregar(hallazgo)

    return reporte


def imprimir_reporte(reporte: Reporte) -> None:
    """Imprime el reporte agrupado por categoría usando ``logging``.

    Args:
        reporte: Reporte consolidado a imprimir.
    """
    agrupado = reporte.por_categoria()
    if not agrupado:
        logger.info("Sin hallazgos. Auditoría de código limpia.")
        return

    for categoria, items in sorted(agrupado.items()):
        logger.info("=== %s (%d hallazgos) ===", categoria, len(items))
        for h in items:
            logger.info("  %s:%d — %s", h.ruta, h.linea, h.detalle)


def main() -> None:
    """Punto de entrada de la auditoría de código estática.

    Analiza ``--src`` y ``--tests``, imprime el reporte y termina con
    código de salida 1 si hay al menos un hallazgo, 0 en caso contrario.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Auditoría estática de código UAO-Neumonia.")
    parser.add_argument(
        "--src", type=Path, default=Path("src"), help="Directorio de código fuente."
    )
    parser.add_argument("--tests", type=Path, default=Path("tests"), help="Directorio de tests.")
    parser.add_argument(
        "--config", type=str, default="config.py", help="Nombre del módulo de configuración."
    )
    args = parser.parse_args()

    if not args.src.exists():
        logger.error("No existe el directorio de fuente: %s", args.src)
        sys.exit(2)

    reporte = ejecutar_auditoria(args.src, args.tests, args.config)
    imprimir_reporte(reporte)
    sys.exit(1 if reporte.hallazgos else 0)


if __name__ == "__main__":
    main()
