"""Loader ligero del inventario oficial XLSX."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


class ErrorInventario(ValueError):
    """Error de resolución que debe aislarse al documento afectado."""


class ResolucionAmbigua(ErrorInventario):
    """El nombre existe en más de una carpeta oficial."""

    def __init__(self, ruta: Path, candidatos: list[dict[str, str]], razon: str) -> None:
        self.ruta = str(ruta)
        self.candidatos = candidatos
        self.razon = razon
        nombres = [
            f"{c.get('Carpeta', '').strip('/')}/{c.get('Nombre estandarizado', '')}"
            for c in candidatos
        ]
        super().__init__(
            f"No hay resolucion oficial unica para {ruta.name}: "
            f"{len(candidatos)} candidatos ({razon}); candidatos={nombres}"
        )


def _celda(celda: ElementTree.Element, shared: list[str]) -> str:
    valor = celda.find("m:v", NS)
    texto = "" if valor is None else valor.text or ""
    return shared[int(texto)] if celda.attrib.get("t") == "s" else texto


def cargar_inventario(ruta: Path) -> list[dict[str, str]]:
    """Carga hoja ``Inventario de Archivos`` con biblioteca estandar."""
    with ZipFile(ruta) as libro:
        shared_root = ElementTree.fromstring(libro.read("xl/sharedStrings.xml"))
        shared = [
            "".join(t.text or "" for t in item.findall(".//m:t", NS))
            for item in shared_root.findall("m:si", NS)
        ]
        workbook = ElementTree.fromstring(libro.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(libro.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        sheet = next(
            item for item in workbook.find("m:sheets", NS)
            if item.attrib["name"] == "Inventario de Archivos"
        )
        hoja = "xl/" + targets[sheet.attrib[REL_ID]]
        root = ElementTree.fromstring(libro.read(hoja))
        filas = [
            [_celda(celda, shared) for celda in fila.findall("m:c", NS)]
            for fila in root.findall(".//m:sheetData/m:row", NS)
        ]
    encabezados = filas[0]
    return [
        {encabezado: valores[indice] if indice < len(valores) else ""
         for indice, encabezado in enumerate(encabezados)}
        for valores in filas[1:]
    ]


def indexar_inventario(registros: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Indexa registros por DOC_ID y rechaza IDs vacios o repetidos."""
    indice: dict[str, dict[str, str]] = {}
    for registro in registros:
        doc_id = registro.get("DOC_ID", "").strip()
        if not doc_id or doc_id in indice:
            raise ValueError(f"DOC_ID oficial invalido o duplicado: {doc_id!r}")
        indice[doc_id] = registro
    return indice


def resolver_archivo(
    ruta: Path,
    inventario: list[dict[str, str]],
    raiz_oficial: Path,
) -> dict[str, str]:
    """Resuelve por ruta oficial, incluyendo una raíz local alternativa.

    La ruta relativa oficial completa tiene prioridad. Si el corpus fue copiado,
    se compara el sufijo ``Carpeta/nombre`` de la ruta local. El nombre aislado
    nunca desambigua documentos repetidos.
    """
    ruta = ruta.resolve()
    raiz_oficial = raiz_oficial.resolve()
    candidatos = [
        registro for registro in inventario
        if registro.get("Nombre estandarizado", "").strip() == ruta.name
    ]
    try:
        relativa = ruta.relative_to(raiz_oficial).as_posix()
    except ValueError:
        relativa = ""
    if relativa:
        exactos = [
            registro for registro in candidatos
            if f"{registro.get('Carpeta', '').strip('/')}/{registro.get('Nombre estandarizado', '').strip()}" == relativa
        ]
        candidatos = exactos or candidatos
    if len(candidatos) != 1 and candidatos:
        sufijos = []
        partes_locales = ruta.parts
        for registro in candidatos:
            partes_oficiales = tuple(
                p for p in (
                    registro.get("Carpeta", "").strip("/"),
                    registro.get("Nombre estandarizado", "").strip(),
                ) if p
            )
            if partes_oficiales and tuple(partes_locales[-len(partes_oficiales):]) == partes_oficiales:
                sufijos.append(registro)
        candidatos = sufijos or candidatos
    if len(candidatos) != 1:
        razon = "nombre inexistente" if not candidatos else "nombre ambiguo o carpeta no coincidente"
        raise ResolucionAmbigua(ruta, candidatos, razon)
    registro = candidatos[0]
    fenomeno = {"F1": 1, "F2": 2, "F3": 3}.get(registro.get("Fenómeno", ""))
    if fenomeno is None:
        raise ErrorInventario(f"Fenomeno oficial invalido para {registro.get('DOC_ID', '')}")
    nombre = registro["Nombre estandarizado"].lower()
    extension = nombre.rsplit(".", 1)[-1] if "." in nombre else ""
    if nombre.endswith(".osm.pbf"):
        extension = "pbf"
    doc_id = registro.get("DOC_ID", "").strip()
    fuente = f"{registro.get('Carpeta', '').strip('/')}/{registro.get('Nombre estandarizado', '').strip()}"
    if not doc_id or not registro.get("Nombre estandarizado", "").strip() or not registro.get("Carpeta", "").strip():
        raise ErrorInventario(f"Metadata oficial incompleta para {doc_id or ruta.name}")
    if not extension or not fuente.split("/")[-1].lower().endswith(extension):
        raise ErrorInventario(f"Formato/fuente oficial incoherentes para {doc_id}")
    return {
        "doc_id": doc_id,
        "fuente": fuente,
        "fenomeno": str(fenomeno),
        "formato": extension,
        "observatorio": registro.get("Observatorio", ""),
        "codigo_observatorio": registro.get("Código Observatorio", ""),
        "carpeta": registro.get("Carpeta", ""),
        "tipo_inventario": registro.get("Tipo", ""),
    }
