"""Loader ligero del inventario oficial XLSX."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


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
    """Resuelve por ruta oficial; corpus prueba usa nombre unico."""
    ruta = ruta.resolve()
    raiz_oficial = raiz_oficial.resolve()
    candidatos = [
        registro for registro in inventario
        if registro.get("Nombre estandarizado", "") == ruta.name
    ]
    try:
        relativa = ruta.relative_to(raiz_oficial).as_posix()
    except ValueError:
        relativa = ""
    if relativa:
        exactos = [
            registro for registro in candidatos
            if f"{registro['Carpeta'].rstrip('/')}/{registro['Nombre estandarizado']}" == relativa
        ]
        candidatos = exactos or candidatos
    if len(candidatos) > 1:
        # Permite procesar una copia del corpus con otra raíz local. La ruta
        # todavía debe conservar la carpeta oficial completa; nombre solo no
        # basta porque el inventario contiene duplicados legítimos.
        ruta_partes = ruta.parts
        por_sufijo = [
            registro for registro in candidatos
            if tuple(registro.get("Carpeta", "").strip("/").split("/"))
            and ruta_partes[-(len(registro["Carpeta"].strip("/").split("/")) + 1):]
            == tuple(registro["Carpeta"].strip("/").split("/")) + (ruta.name,)
        ]
        candidatos = por_sufijo or candidatos
    if len(candidatos) != 1:
        raise ValueError(f"No hay resolucion oficial unica para {ruta.name}: {len(candidatos)} candidatos")
    registro = candidatos[0]
    fenomeno = {"F1": 1, "F2": 2, "F3": 3}.get(registro.get("Fenómeno", ""))
    if fenomeno is None:
        raise ValueError(f"Fenomeno oficial invalido para {registro['DOC_ID']}")
    nombre = registro["Nombre estandarizado"].lower()
    extension = nombre.rsplit(".", 1)[-1] if "." in nombre else ""
    if nombre.endswith(".osm.pbf"):
        extension = "pbf"
    return {
        "doc_id": registro["DOC_ID"],
        "fuente": f"{registro['Carpeta'].strip('/')}/{registro['Nombre estandarizado']}",
        "fenomeno": str(fenomeno),
        "formato": extension,
        "observatorio": registro.get("Observatorio", ""),
        "codigo_observatorio": registro.get("Código Observatorio", ""),
        "carpeta": registro.get("Carpeta", ""),
        "tipo_inventario": registro.get("Tipo", ""),
    }
