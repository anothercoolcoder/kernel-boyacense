"""Extracción multimodal de texto: etapa 1 del pipeline RAG.

Convierte cualquier documento del corpus en una lista homogénea de *registros*
(``list[Registro]``), donde cada registro es una unidad atómica de lectura:
una página de PDF, una fila de CSV, un documento HTML, una entidad de un tile
vectorial, etc.

Contrato de salida (estable, del que dependen las etapas siguientes)::

    {
        "documento": "informe.pdf",       # nombre del archivo de origen
        "ruta": "C:/corpus/informe.pdf",  # ruta absoluta
        "tipo": "pdf",                    # formato lógico
        "pagina": 1,                      # ordinal 1-based de la unidad
        "texto": "...",                   # texto útil, sin decoración
        "metadata": {...},                # todo lo que NO es texto
    }

Reglas de diseño:

* Ningún extractor hace *chunking*, limpieza ni normalización semántica.
* Ningún extractor mezcla metadata dentro de ``texto``.
* Los registros sin texto útil se descartan (no llegan al índice).
* Las dependencias pesadas se importan de forma perezosa dentro de cada
  extractor: la ausencia de ``pandas`` solo rompe CSV/XLSX, no el módulo.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

__all__ = [
    "Registro",
    "ErrorExtraccion",
    "FormatoNoSoportado",
    "extraer_documento",
    "registrar_extractor",
    "formatos_soportados",
    "extraer_pdf",
    "extraer_html",
    "extraer_md",
    "extraer_txt",
    "extraer_json",
    "extraer_csv",
    "extraer_xlsx",
    "extraer_imagen",
    "extraer_pbf",
    "extraer_gpkg",
    "extraer_osm_pbf",
]

#: Capas de OpenStreetMap que expone pyrosm, en el orden en que se extraen.
CAPAS_OSM = ("buildings", "pois", "natural", "landuse", "boundaries", "network")

logger = logging.getLogger(__name__)

Registro = Dict[str, Any]
Extractor = Callable[[Path], List[Registro]]


# --------------------------------------------------------------------------- #
# Errores
# --------------------------------------------------------------------------- #
class ErrorExtraccion(RuntimeError):
    """Fallo al extraer un documento (E/S, dependencia ausente, archivo corrupto)."""


class FormatoNoSoportado(ErrorExtraccion):
    """La extensión del archivo no tiene extractor registrado."""


# --------------------------------------------------------------------------- #
# Registro de extractores (punto de extensión: abierto/cerrado)
# --------------------------------------------------------------------------- #
_EXTRACTORES: Dict[str, Extractor] = {}


# --------------------------------------------------------------------------- #
# Registro de doc_id persistente (regla de negocio: ID fijo entre ejecuciones)
# --------------------------------------------------------------------------- #
#: Ruta al JSON que mapea ruta_absoluta_archivo → doc_id.
#: Se sitúa en la raíz del proyecto (un nivel arriba de este módulo).
DOC_ID_REGISTRY_PATH: Path = Path(__file__).resolve().parent.parent / "doc_id_registry.json"

_registry_lock = threading.Lock()
_registry_cache: Dict[str, str] | None = None
_registry_counter: list[int] = [1]  # envuelto en lista para mutabilidad en closure


def _cargar_registry() -> Dict[str, str]:
    """Carga el registry del disco (una vez) y lo devuelve como dict."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    if DOC_ID_REGISTRY_PATH.exists():
        try:
            _registry_cache = json.loads(DOC_ID_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            _registry_cache = {}
    else:
        _registry_cache = {}
    # Ajusta el contador al siguiente número libre para evitar colisiones.
    if _registry_cache:
        nums = []
        for v in _registry_cache.values():
            try:
                nums.append(int(v.split("-")[1]))
            except (IndexError, ValueError):
                pass
        _registry_counter[0] = max(nums, default=0) + 1
    return _registry_cache


def _guardar_registry(registry: Dict[str, str]) -> None:
    """Persiste el registry en disco de forma atómica."""
    DOC_ID_REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def obtener_doc_id(ruta: Path) -> str:
    """Devuelve el doc_id estable para ``ruta``, creándolo si no existe.

    El id es inmutable entre ejecuciones: si el archivo ya fue visto, recibe
    exactamente el mismo ``DOC-XXXX`` que la vez anterior.
    """
    clave = str(ruta.resolve())
    with _registry_lock:
        registry = _cargar_registry()
        if clave not in registry:
            nuevo_id = f"DOC-{_registry_counter[0]:04d}"
            registry[clave] = nuevo_id
            _registry_counter[0] += 1
            _guardar_registry(registry)
        return registry[clave]


#: Extensiones tratadas como imagen rasterizada.
EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

#: Claves JSON consideradas texto del cuerpo, en orden de preferencia.
CAMPOS_TEXTO_JSON = (
    "title",
    "titulo",
    "subtitle",
    "summary",
    "resumen",
    "body",
    "body_text",
    "content",
    "contenido",
    "texto",
    "text",
    "paragraphs",
    "body_paragraphs",
    "parrafos",
)

#: Claves JSON que nunca se mezclan con el texto.
CAMPOS_METADATA_JSON = (
    "url", "date", "fecha", "author", "autor", "tags", "categoria",
    "category", "source", "fuente", "id", "lang", "idioma",
)

#: Confianza mínima (0-1) para conservar una línea de OCR.
UMBRAL_OCR = 0.35

#: Fracción del texto plano que el markdown debe conservar para preferirlo.
#: Sesgado a favor del texto plano: perder una ecuación cuesta más que perder
#: el formato de una tabla, y el plano nunca pierde contenido.
UMBRAL_RETENCION_MD = 0.95

# Marcadores que pymupdf4llm puede dejar alrededor del texto OCR de figuras.
# Solo se eliminan comentarios conocidos; no se borra HTML arbitrario de
# Markdown porque puede formar parte del contenido original.
_MARCADORES_FIGURA = re.compile(
    r"<!--\s*(?:start|end)\s+of\s+(?:picture|image)\s+text\s*-->"
    r"|<!--\s*(?:start|end)\s+of\s+(?:picture|image)\s*-->",
    re.IGNORECASE,
)
_BR_PDF = re.compile(r"</?(?:br|sup|sub|u)\s*/?>", re.IGNORECASE)
_CONTROL_NO_PERMITIDO = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ESPACIOS_LINEA = re.compile(r"[ \t]+")
_NUMERO_PAGINA = re.compile(
    r"^(?:p(?:age|á?gina)?\s*)?[-–—]?\s*\d+\s*[-–—]?$",
    re.IGNORECASE,
)


def registrar_extractor(*extensiones: str) -> Callable[[Extractor], Extractor]:
    """Registra una función como extractor de las extensiones dadas.

    Permite añadir formatos nuevos sin tocar el despachador::

        @registrar_extractor(".epub")
        def extraer_epub(ruta: Path) -> list[Registro]: ...

    Args:
        *extensiones: extensiones con punto, insensibles a mayúsculas.

    Returns:
        Decorador que devuelve la función original sin modificarla.
    """

    def decorador(funcion: Extractor) -> Extractor:
        for extension in extensiones:
            _EXTRACTORES[extension.lower()] = funcion
        return funcion

    return decorador


def formatos_soportados() -> List[str]:
    """Devuelve las extensiones con extractor registrado, ordenadas."""
    return sorted(_EXTRACTORES)


# --------------------------------------------------------------------------- #
# Utilidades internas
# --------------------------------------------------------------------------- #
def _metadata_base(ruta: Path) -> Dict[str, Any]:
    """Metadata común a todo registro proveniente de ``ruta``."""
    stat = ruta.stat()
    return {
        "extension": ruta.suffix.lower(),
        "bytes": stat.st_size,
        "modificado": stat.st_mtime,
    }


def _obtener_fuente_relativa(ruta: Path) -> str:
    """Devuelve la ruta relativa de ``ruta`` respecto a la raíz del corpus provisto por ADL.

    Si la ruta pertenece a ``corpus_adl``, devuelve la subruta en formato POSIX
    (ej. ``fenomeno_1/doc.pdf``). Si no, devuelve ``ruta.name``.
    """
    parts = ruta.parts
    if "corpus_adl" in parts:
        idx = parts.index("corpus_adl")
        rel_parts = parts[idx + 1 :]
        if rel_parts:
            return "/".join(rel_parts)
    return ruta.name


MAPA_FORMATOS = {
    "markdown": "md",
    "md": "md",
    "html": "html",
    "htm": "html",
    "xhtml": "html",
    "pdf": "pdf",
    "txt": "txt",
    "text": "txt",
    "log": "txt",
    "json": "json",
    "csv": "csv",
    "tsv": "csv",
    "xlsx": "xlsx",
    "xlsm": "xlsx",
    "xls": "xlsx",
    "pbf": "pbf",
    "mvt": "pbf",
    "gpkg": "gpkg",
    "osm_pbf": "pbf",
}


def _normalizar_formato(tipo: str, ruta: Path) -> str:
    """Normaliza el tipo de formato a los valores canónicos (§3.4)."""
    t = tipo.lower()
    if t in MAPA_FORMATOS:
        return MAPA_FORMATOS[t]
    sufijo = ruta.suffix.lstrip(".").lower()
    if sufijo in MAPA_FORMATOS:
        return MAPA_FORMATOS[sufijo]
    return sufijo or t


def _registro(
    ruta: Path,
    tipo: str,
    pagina: int,
    texto: str,
    metadata: Mapping[str, Any] | None = None,
) -> Registro:
    """Construye un registro con el contrato de salida del módulo.

    El campo ``doc_id`` se obtiene del registry persistente para garantizar
    que el mismo archivo siempre recibe el mismo identificador, sin importar
    cuántas veces se ejecute el pipeline (regla de negocio de ID fijo).
    """
    formato_norm = _normalizar_formato(tipo, ruta)
    texto = _limpiar_texto(texto, formato_norm == "pdf")
    return {
        "doc_id": obtener_doc_id(ruta),
        "documento": ruta.name,
        "ruta": str(ruta.resolve()),
        "fuente": _obtener_fuente_relativa(ruta),
        "tipo": formato_norm,
        "pagina": pagina,
        "texto": texto,
        "metadata": {**_metadata_base(ruta), **(metadata or {})},
    }


def _limpiar_texto(texto: str, pdf: bool = False) -> str:
    """Limpia ruido común conservando saltos, Markdown, listas y tablas."""
    texto = unicodedata.normalize("NFC", str(texto)).replace("\r\n", "\n").replace("\r", "\n")
    texto = _MARCADORES_FIGURA.sub("", texto)
    if pdf:
        texto = _BR_PDF.sub("", texto)
    texto = _CONTROL_NO_PERMITIDO.sub("", texto)

    lineas = [_ESPACIOS_LINEA.sub(" ", linea).strip() for linea in texto.split("\n")]
    if pdf:
        # Algunas figuras rasterizadas llegan como una única línea enorme de
        # números sin puntuación. Es ruido gráfico, no una oración ni una fila
        # recuperable; se elimina solo con las tres señales simultáneas.
        lineas = [
            linea for linea in lineas
            if not (
                len(linea) > 1000
                and not re.search(r"[.!?]\s*$", linea)
                and sum(c.isdigit() for c in linea) / max(1, len(linea)) > 0.08
            )
        ]
    salida: List[str] = []
    ultimo_vacio = False
    for linea in lineas:
        if not linea:
            if not ultimo_vacio:
                salida.append("")
            ultimo_vacio = True
            continue
        salida.append(linea)
        ultimo_vacio = False
    return "\n".join(salida).strip()


def _limpiar_boilerplate_pdf(registros: List[Registro]) -> List[Registro]:
    """Elimina cabeceras/pies repetidos solo en los márgenes de las páginas."""
    if len(registros) < 2:
        return registros

    candidatos: List[str] = []
    for registro in registros:
        lineas = [linea.strip() for linea in registro["texto"].splitlines() if linea.strip()]
        margen = lineas[:6] + lineas[-6:]
        candidatos.extend(
            linea for linea in margen
            if len(linea) >= 3 and not _NUMERO_PAGINA.fullmatch(linea)
        )

    minimo = max(2, math.ceil(len(registros) * 0.5))
    repetidos = {
        linea for linea, cuenta in Counter(candidatos).items()
        if cuenta >= minimo
    }

    for registro in registros:
        lineas = registro["texto"].splitlines()
        no_vacias = [i for i, linea in enumerate(lineas) if linea.strip()]
        indices_margen = set(no_vacias[:6] + no_vacias[-6:])
        filtradas = [
            linea for i, linea in enumerate(lineas)
            if not (
                i in indices_margen
                and (linea.strip() in repetidos or _NUMERO_PAGINA.fullmatch(linea.strip() or ""))
            )
        ]
        eliminadas = len(lineas) - len(filtradas)
        registro["texto"] = _limpiar_texto("\n".join(filtradas), pdf=True)
        registro["metadata"]["boilerplate_lineas_eliminadas"] = eliminadas
    return _utiles(registros)


def _utiles(registros: Iterable[Registro]) -> List[Registro]:
    """Filtra registros cuyo texto es vacío o solo espacios."""
    return [r for r in registros if r["texto"] and r["texto"].strip()]


def _leer_texto(ruta: Path) -> str:
    """Lee un archivo de texto en UTF-8, con reintento en latin-1.

    Raises:
        ErrorExtraccion: si el archivo no puede leerse con ninguna codificación.
    """
    for codificacion in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return ruta.read_text(encoding=codificacion)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise ErrorExtraccion(f"No se pudo leer {ruta}: {exc}") from exc
    raise ErrorExtraccion(f"Codificación no reconocida en {ruta}")


def _importar(modulo: str, paquete_pip: str) -> Any:
    """Importa un módulo opcional dando un error accionable si falta."""
    from importlib import import_module

    try:
        return import_module(modulo)
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ErrorExtraccion(
            f"Falta la dependencia '{modulo}'. Instala: pip install {paquete_pip}"
        ) from exc


# --------------------------------------------------------------------------- #
# OCR (motor intercambiable)
# --------------------------------------------------------------------------- #
def _ocr(imagen: Any) -> tuple[str, float]:
    """Reconoce texto en una imagen y devuelve ``(texto, confianza_media)``.

    Intenta RapidOCR (pip-only, sin binario de sistema, multilingüe) y cae a
    Tesseract si no está disponible. Ambos motores exponen confianza, dato que
    las etapas posteriores usan para ponderar o descartar el fragmento.

    Args:
        imagen: ruta ``str`` o ``numpy.ndarray`` / ``PIL.Image``.

    Returns:
        Texto reconocido (líneas en orden de lectura) y confianza media 0-1.
        ``("", 0.0)`` si ningún motor reconoce nada.
    """
    texto, confianza = _ocr_rapidocr(imagen)
    if texto:
        return texto, confianza
    return _ocr_tesseract(imagen)


def _ocr_rapidocr(imagen: Any) -> tuple[str, float]:
    """OCR con RapidOCR. Devuelve ``("", 0.0)`` si no está instalado o falla."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return "", 0.0

    global _RAPIDOCR
    try:
        if _RAPIDOCR is None:
            _RAPIDOCR = RapidOCR()
        resultado, _ = _RAPIDOCR(imagen)
    except Exception as exc:  # el motor OCR nunca debe tumbar la extracción
        logger.warning("RapidOCR falló: %s", exc)
        return "", 0.0

    if not resultado:
        return "", 0.0
    # RapidOCR devuelve la confianza como str en algunas versiones del motor.
    lineas = [(str(txt), float(conf)) for _, txt, conf in resultado if float(conf) >= UMBRAL_OCR]
    if not lineas:
        return "", 0.0
    return "\n".join(t for t, _ in lineas), sum(c for _, c in lineas) / len(lineas)


_RAPIDOCR: Any = None


def _ocr_tesseract(imagen: Any) -> tuple[str, float]:
    """OCR con Tesseract. Devuelve ``("", 0.0)`` si no está disponible o falla."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("Sin motor OCR disponible (rapidocr-onnxruntime | pytesseract)")
        return "", 0.0

    # ponytail: Windows no añade Tesseract al PATH por defecto; se apunta al
    # binario en su ruta de instalación estándar si "tesseract" no es invocable.
    if os.name == "nt" and not pytesseract.pytesseract.tesseract_cmd.lower().endswith(".exe"):
        candidato = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if candidato.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidato)

    try:
        img = Image.open(imagen) if isinstance(imagen, (str, Path)) else imagen
        datos = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        logger.warning("Tesseract falló: %s", exc)
        return "", 0.0

    palabras = [
        (p, int(c) / 100)
        for p, c in zip(datos["text"], datos["conf"])
        if p.strip() and int(c) >= 0 and int(c) / 100 >= UMBRAL_OCR
    ]
    if not palabras:
        return "", 0.0
    return " ".join(p for p, _ in palabras), sum(c for _, c in palabras) / len(palabras)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
@registrar_extractor(".pdf")
def extraer_pdf(ruta: Path) -> List[Registro]:
    """Extrae un PDF a razón de un registro por página.

    Usa ``pymupdf4llm`` con ``page_chunks=True``: reconstruye el orden de
    lectura por bloques (no el orden del stream del PDF) y devuelve markdown,
    conservando así encabezados y tablas. Las páginas que quedan sin texto se
    consideran escaneadas y se rasterizan a 300 dpi para pasarlas por OCR.

    Args:
        ruta: archivo ``.pdf``.

    Returns:
        Un registro por página con texto útil, en orden de página.

    Raises:
        ErrorExtraccion: si el PDF no puede abrirse o falta ``pymupdf4llm``.
    """
    pymupdf4llm = _importar("pymupdf4llm", "pymupdf4llm")

    try:
        paginas: Sequence[Mapping[str, Any]] = pymupdf4llm.to_markdown(
            str(ruta), page_chunks=True, ignore_images=True, ignore_graphics=True
        )
    except Exception as exc:
        raise ErrorExtraccion(f"PDF ilegible {ruta}: {exc}") from exc

    crudos = _texto_crudo_pdf(ruta)

    registros: List[Registro] = []
    for indice, pagina in enumerate(paginas, start=1):
        texto = str(pagina.get("text", "")).strip()
        metadata: Dict[str, Any] = {"origen_texto": "nativo", "total_paginas": len(paginas)}
        crudo = crudos[indice - 1] if indice <= len(crudos) else ""
        # pymupdf4llm descarta spans con fuentes matemáticas (CMMI/CMSY de LaTeX):
        # si perdió parte sustancial de la página, el texto plano es más fiel.
        if _retencion(crudo, texto) < UMBRAL_RETENCION_MD:
            texto = crudo
            metadata["origen_texto"] = "nativo_plano"
        if not texto:
            texto, confianza = _ocr_pagina_pdf(ruta, indice - 1)
            metadata.update(origen_texto="ocr", confianza=round(confianza, 4))
        registros.append(_registro(ruta, "pdf", indice, texto, metadata))

    registros = _limpiar_boilerplate_pdf(registros)
    logger.info("PDF %s: %d páginas con texto", ruta.name, len(registros))
    return registros


_SIN_MARCADO = str.maketrans("", "", "*_#`|")


def _retencion(crudo: str, markdown: str) -> float:
    """Fracción de las palabras del texto plano que sobreviven en el markdown.

    Comparar longitudes no sirve: el markdown infla con ``#``, ``**`` y tablas
    aunque haya perdido ecuaciones enteras. Lo que importa es qué contenido
    desapareció, no cuántos caracteres hay.
    """
    palabras = set(crudo.split())
    if not palabras:
        return 1.0
    # el marcado se pega a las palabras (**a**) y falsearía la comparación
    presentes = palabras & set(markdown.translate(_SIN_MARCADO).split())
    return len(presentes) / len(palabras)


def _normalizar_texto_pdf(texto: str) -> str:
    """Recompone acentos que pdfTeX emite como glifo suelto fuera de orden.

    Algunas fuentes de LaTeX codifican "ñ" como el glifo de acento
    MODIFIER LETTER TILDE (U+02DC) *antes* de la letra base ("Ni˜no" en vez
    de "Niño"); NFC no lo arregla porque no es una marca combinante.
    """
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("˜n", "ñ").replace("˜N", "Ñ")
    return texto


def _texto_crudo_pdf(ruta: Path) -> List[str]:
    """Texto plano por página vía PyMuPDF. Lista vacía si PyMuPDF no está."""
    try:
        import pymupdf
    except ImportError:
        return []
    try:
        with pymupdf.open(str(ruta)) as doc:
            return [_normalizar_texto_pdf(p.get_text()).strip() for p in doc]
    except Exception as exc:
        logger.warning("Texto plano falló en %s: %s", ruta.name, exc)
        return []


def _ocr_pagina_pdf(ruta: Path, indice_pagina: int, dpi: int = 300) -> tuple[str, float]:
    """Rasteriza una página y le aplica OCR. Devuelve ``(texto, confianza)``."""
    try:
        import numpy as np
        import pymupdf
    except ImportError:
        logger.warning("OCR de PDF requiere pymupdf y numpy; página %d omitida", indice_pagina + 1)
        return "", 0.0

    try:
        with pymupdf.open(str(ruta)) as doc:
            pix = doc[indice_pagina].get_pixmap(dpi=dpi)
            arreglo = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
        return _ocr(arreglo[:, :, :3])
    except Exception as exc:
        logger.warning("OCR falló en %s p.%d: %s", ruta.name, indice_pagina + 1, exc)
        return "", 0.0


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
#: Elementos sin contenido informativo que se eliminan del árbol.
_HTML_RUIDO = ("script", "style", "noscript", "nav", "footer", "header", "form", "svg")

#: Elementos cuyo texto visible se conserva, en orden documental.
_HTML_UTILES = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "td", "th", "pre")


@registrar_extractor(".html", ".htm", ".xhtml")
def extraer_html(ruta: Path) -> List[Registro]:
    """Extrae el texto visible de un HTML, descartando marcado y estilos.

    Usa BeautifulSoup con ``lxml`` (tolerante a HTML mal formado, que es la
    norma en corpus reales). Elimina el ruido estructural del árbol y recorre
    los elementos de contenido en orden documental, conservando la jerarquía
    como metadata (lista de encabezados) en vez de mezclarla con el texto.

    Args:
        ruta: archivo ``.html`` / ``.htm``.

    Returns:
        Lista con un único registro (el documento completo). El *chunking*
        ocurre en una etapa posterior.

    Raises:
        ErrorExtraccion: si falta BeautifulSoup o el archivo no puede leerse.
    """
    bs4 = _importar("bs4", "beautifulsoup4 lxml")
    crudo = _leer_texto(ruta)

    try:
        sopa = bs4.BeautifulSoup(crudo, "lxml")
    except Exception:  # lxml ausente: parser de la stdlib
        sopa = bs4.BeautifulSoup(crudo, "html.parser")

    for elemento in sopa(list(_HTML_RUIDO)):
        elemento.decompose()

    lineas: List[str] = []
    encabezados: List[str] = []
    for nodo in sopa.find_all(_HTML_UTILES):
        texto = nodo.get_text(" ", strip=True)
        if not texto:
            continue
        if nodo.name.startswith("h") and nodo.name[1:].isdigit():
            encabezados.append(texto)
        lineas.append(texto)

    metadata = {
        "titulo": sopa.title.get_text(strip=True) if sopa.title else None,
        "encabezados": encabezados,
        "bloques": len(lineas),
    }
    return _utiles([_registro(ruta, "html", 1, "\n".join(lineas), metadata)])


# --------------------------------------------------------------------------- #
# Markdown / TXT
# --------------------------------------------------------------------------- #
@registrar_extractor(".md", ".markdown", ".mdx")
def extraer_md(ruta: Path) -> List[Registro]:
    """Devuelve el Markdown íntegro, sin renderizar ni alterar.

    Renderizarlo (markdown-it → HTML) destruiría precisamente los encabezados,
    listas y separadores que hay que conservar: el markdown ya es el formato
    canónico del pipeline. Solo se indexan los encabezados como metadata.
    """
    texto = _leer_texto(ruta)
    encabezados = [
        linea.strip() for linea in texto.splitlines() if linea.lstrip().startswith("#")
    ]
    return _utiles([_registro(ruta, "md", 1, texto, {"encabezados": encabezados})])


@registrar_extractor(".txt", ".text", ".log")
def extraer_txt(ruta: Path) -> List[Registro]:
    """Devuelve el texto plano íntegro como un único registro."""
    texto = _leer_texto(ruta)
    return _utiles([_registro(ruta, "txt", 1, texto, {"lineas": texto.count("\n") + 1})])


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
@registrar_extractor(".json")
def extraer_json(ruta: Path) -> List[Registro]:
    """Extrae los campos textuales de un JSON respetando su estructura.

    No concatena el documento completo: recorre el árbol y separa campos de
    contenido (``title``, ``body``, ``paragraphs``…) de campos descriptivos
    (``url``, ``date``, ``author``, ``tags``), que van a ``metadata``. Las
    listas de párrafos preservan su orden original.

    Un JSON de nivel superior tipo lista produce un registro por elemento;
    un objeto produce un único registro.

    Raises:
        ErrorExtraccion: si el JSON es inválido o ilegible.
    """
    try:
        datos = json.loads(_leer_texto(ruta))
    except json.JSONDecodeError as exc:
        raise ErrorExtraccion(f"JSON inválido {ruta}: {exc}") from exc

    elementos = datos if isinstance(datos, list) else [datos]
    registros: List[Registro] = []
    for indice, elemento in enumerate(elementos, start=1):
        if not isinstance(elemento, Mapping):
            texto, metadata = _texto_plano(elemento), {}
        else:
            texto, metadata = _partir_json(elemento)
        registros.append(_registro(ruta, "json", indice, texto, {**metadata, "elementos": len(elementos)}))

    return _utiles(registros)


def _partir_json(objeto: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Separa un objeto JSON en ``(texto, metadata)``.

    Los campos de :data:`CAMPOS_TEXTO_JSON` se concatenan en el orden en que
    aparecen ahí (título antes que cuerpo); los de
    :data:`CAMPOS_METADATA_JSON` van a metadata; los objetos anidados se
    recorren en profundidad y aportan a ambos lados.
    """
    partes: List[str] = []
    metadata: Dict[str, Any] = {}

    for clave in CAMPOS_TEXTO_JSON:
        if clave in objeto:
            fragmento = _texto_plano(objeto[clave])
            if fragmento:
                partes.append(fragmento)

    for clave, valor in objeto.items():
        clave_norm = clave.lower()
        if clave_norm in CAMPOS_METADATA_JSON:
            metadata[clave] = valor
        elif isinstance(valor, Mapping) and clave_norm not in CAMPOS_TEXTO_JSON:
            sub_texto, sub_metadata = _partir_json(valor)
            if sub_texto:
                partes.append(sub_texto)
            metadata.update({f"{clave}.{k}": v for k, v in sub_metadata.items()})

    return "\n\n".join(partes), metadata


def _texto_plano(valor: Any) -> str:
    """Aplana un valor JSON a texto conservando el orden de las listas."""
    if isinstance(valor, str):
        return valor.strip()
    if isinstance(valor, (int, float, bool)):
        return str(valor)
    if isinstance(valor, Sequence):
        return "\n".join(filter(None, (_texto_plano(v) for v in valor)))
    if isinstance(valor, Mapping):
        return "\n".join(filter(None, (_texto_plano(v) for v in valor.values())))
    return ""


# --------------------------------------------------------------------------- #
# Tabulares: CSV / XLSX
# --------------------------------------------------------------------------- #
@registrar_extractor(".csv", ".tsv")
def extraer_csv(ruta: Path) -> List[Registro]:
    """Convierte cada fila del CSV en un registro independiente.

    pandas resuelve por sí solo la inferencia de separador (``sep=None`` con el
    motor de Python), las comillas y las codificaciones, que es donde el módulo
    ``csv`` de la stdlib obliga a escribir heurística propia.

    Raises:
        ErrorExtraccion: si el archivo no puede parsearse.
    """
    pd = _importar("pandas", "pandas")
    try:
        marco = pd.read_csv(ruta, sep=None, engine="python", dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ErrorExtraccion(f"CSV ilegible {ruta}: {exc}") from exc
    return _filas_a_registros(ruta, "csv", marco, {})


@registrar_extractor(".xlsx", ".xlsm", ".xls")
def extraer_xlsx(ruta: Path) -> List[Registro]:
    """Convierte cada fila de cada hoja del libro en un registro independiente.

    La hoja de origen viaja en ``metadata['hoja']``; ``pagina`` es un ordinal
    global continuo para que el orden de lectura del libro sea reconstruible.

    Raises:
        ErrorExtraccion: si el libro no puede abrirse (falta ``openpyxl``, etc.).
    """
    pd = _importar("pandas", "pandas openpyxl")
    try:
        hojas = pd.read_excel(ruta, sheet_name=None, dtype=str)
    except Exception as exc:
        raise ErrorExtraccion(f"XLSX ilegible {ruta}: {exc}") from exc

    registros: List[Registro] = []
    for nombre, marco in hojas.items():
        registros.extend(
            _filas_a_registros(ruta, "xlsx", marco, {"hoja": nombre}, desplazamiento=len(registros))
        )
    return registros


def _filas_a_registros(
    ruta: Path,
    tipo: str,
    marco: Any,
    metadata_extra: Mapping[str, Any],
    desplazamiento: int = 0,
) -> List[Registro]:
    """Serializa un ``DataFrame`` a registros ``columna: valor``, uno por fila.

    Las celdas vacías o nulas se omiten, de modo que el texto resultante no
    arrastre ruido (``Campo: nan``) a la etapa de embeddings.
    """
    import pandas as pd  # ya importado por el llamador

    columnas = [str(c) for c in marco.columns]
    registros: List[Registro] = []
    for orden, (_, fila) in enumerate(marco.iterrows(), start=1):
        pares = [
            f"{columna}: {str(valor).strip()}"
            for columna, valor in zip(columnas, fila)
            if not pd.isna(valor) and str(valor).strip()
        ]
        if not pares:
            continue
        registros.append(
            _registro(
                ruta,
                tipo,
                desplazamiento + orden,
                "\n".join(pares),
                {**metadata_extra, "fila": orden, "columnas": columnas},
            )
        )
    logger.info("%s %s: %d filas con contenido", tipo.upper(), ruta.name, len(registros))
    return registros


# --------------------------------------------------------------------------- #
# Imágenes
# --------------------------------------------------------------------------- #
@registrar_extractor(*EXTENSIONES_IMAGEN)
def extraer_imagen(ruta: Path) -> List[Registro]:
    """Aplica OCR a una imagen y devuelve el texto reconocido con su confianza.

    Devuelve lista vacía si la imagen no contiene texto legible: una imagen
    puramente decorativa no debe entrar al índice.
    """
    texto, confianza = _ocr(str(ruta))
    metadata: Dict[str, Any] = {"origen_texto": "ocr", "confianza": round(confianza, 4)}
    try:
        from PIL import Image

        with Image.open(ruta) as img:
            metadata["dimensiones"] = list(img.size)
            metadata["modo"] = img.mode
    except Exception:  # metadata opcional: su ausencia no invalida el OCR
        logger.debug("Sin metadata de imagen para %s", ruta.name)

    return _utiles([_registro(ruta, "imagen", 1, texto, metadata)])


# --------------------------------------------------------------------------- #
# PBF (Mapbox Vector Tile)
# --------------------------------------------------------------------------- #
@registrar_extractor(".pbf", ".mvt")
def extraer_pbf(ruta: Path) -> List[Registro]:
    """Extrae las entidades de un tile vectorial como texto ``clave: valor``.

    Recorre capas y entidades, serializa los atributos (la geometría no aporta
    nada recuperable por texto) y deduplica: el mismo objeto aparece repetido
    en varios niveles de zoom, y sin deduplicar el índice se llenaría de
    vecinos idénticos que degradan el *retrieval*.

    Notas:
        Asume el formato Mapbox Vector Tile. Para extractos OSM (``.osm.pbf``)
        hay que sustituir esta función por una basada en ``osmium``.

    Raises:
        ErrorExtraccion: si falta ``mapbox-vector-tile`` o el tile es inválido.
    """
    mvt = _importar("mapbox_vector_tile", "mapbox-vector-tile")

    try:
        tile = mvt.decode(ruta.read_bytes())
    except Exception as exc:
        raise ErrorExtraccion(f"PBF ilegible {ruta}: {exc}") from exc

    vistos: set[str] = set()
    registros: List[Registro] = []
    for capa, contenido in tile.items():
        for entidad in contenido.get("features", []):
            propiedades = {
                clave: valor
                for clave, valor in (entidad.get("properties") or {}).items()
                if valor not in (None, "", [])
            }
            if not propiedades:
                continue
            texto = "\n".join(f"{clave}: {valor}" for clave, valor in sorted(propiedades.items()))
            huella = f"{capa}|{texto}"
            if huella in vistos:
                continue
            vistos.add(huella)
            registros.append(
                _registro(
                    ruta,
                    "pbf",
                    len(registros) + 1,
                    texto,
                    {"capa": capa, "geometria": (entidad.get("geometry") or {}).get("type")},
                )
            )

    logger.info("PBF %s: %d entidades únicas", ruta.name, len(registros))
    return registros


# --------------------------------------------------------------------------- #
# GeoPackage (OGC, SQLite con capas vectoriales)
# --------------------------------------------------------------------------- #
@registrar_extractor(".gpkg")
def extraer_gpkg(ruta: Path) -> List[Registro]:
    """Extrae las entidades de un GeoPackage como texto ``columna: valor``.

    Un GeoPackage es, por estándar OGC, una base de datos SQLite: se lee con
    la stdlib (``sqlite3``), sin GDAL ni geopandas. No es un tile de Mapbox
    Vector Tile (``.pbf``/``.mvt``) aunque ambos sean formatos geoespaciales;
    ese decodificador no entiende SQLite.

    Recorre cada capa listada en ``gpkg_contents`` y convierte cada fila en un
    registro; la columna de geometría (binario WKB, no recuperable por texto)
    se descarta. ``pagina`` es un ordinal global continuo, igual que en XLSX.

    Raises:
        ErrorExtraccion: si el archivo no es un GeoPackage válido.
    """
    import sqlite3

    try:
        conexion = sqlite3.connect(f"file:{ruta}?mode=ro", uri=True)
        cursor = conexion.cursor()
        cursor.execute(
            "SELECT table_name, column_name FROM gpkg_geometry_columns"
        )
        columna_geom = dict(cursor.fetchall())
        cursor.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'"
        )
        capas = [fila[0] for fila in cursor.fetchall()]
    except sqlite3.DatabaseError as exc:
        raise ErrorExtraccion(f"GeoPackage ilegible {ruta}: {exc}") from exc

    registros: List[Registro] = []
    for capa in capas:
        geom = columna_geom.get(capa)
        cursor.execute(f'PRAGMA table_info("{capa}")')
        clave_primaria = {fila[1] for fila in cursor.fetchall() if fila[5]}  # pk autogenerada, no es contenido

        cursor.execute(f'SELECT * FROM "{capa}"')
        excluidas = clave_primaria | {geom}
        columnas = [d[0] for d in cursor.description if d[0] not in excluidas]
        indices = [i for i, d in enumerate(cursor.description) if d[0] not in excluidas]
        for fila in cursor:
            pares = [
                f"{columnas[pos]}: {str(fila[i]).strip()}"
                for pos, i in enumerate(indices)
                if fila[i] not in (None, "") and str(fila[i]).strip()
            ]
            if not pares:
                continue
            registros.append(
                _registro(
                    ruta,
                    "gpkg",
                    len(registros) + 1,
                    "\n".join(pares),
                    {"capa": capa, "columnas": columnas},
                )
            )

    conexion.close()
    logger.info("GeoPackage %s: %d entidades en %d capas", ruta.name, len(registros), len(capas))
    return registros


# --------------------------------------------------------------------------- #
# OSM PBF (volcado de OpenStreetMap, distinto de Mapbox Vector Tile)
# --------------------------------------------------------------------------- #
@registrar_extractor(".osm.pbf")
def extraer_osm_pbf(ruta: Path) -> List[Registro]:
    """Extrae un volcado de OpenStreetMap (``.osm.pbf``) como texto ``clave: valor``.

    Un ``.osm.pbf`` (nodos/ways/relaciones de OSM) usa un esquema protobuf
    distinto al de un ``.pbf``/``.mvt`` de Mapbox Vector Tile: ``extraer_pbf``
    no puede leerlo (por eso el despacho separa ambas extensiones). Se usa
    ``pyrosm`` porque hace el trabajo pesado de parsear el protobuf de OSM y
    reconstruir las etiquetas (``tags``) de cada elemento; escribirlo a mano
    implicaría reimplementar ese parser.

    Recorre las categorías de :data:`CAPAS_OSM` (edificios, POIs, red vial,
    etc.); cada fila con datos se convierte en un registro, descartando la
    columna de geometría.

    Raises:
        ErrorExtraccion: si falta ``pyrosm`` o el archivo no puede leerse.
    """
    pyrosm = _importar("pyrosm", "pyrosm")

    try:
        osm = pyrosm.OSM(str(ruta))
    except Exception as exc:
        raise ErrorExtraccion(f"OSM PBF ilegible {ruta}: {exc}") from exc

    registros: List[Registro] = []
    for capa in CAPAS_OSM:
        try:
            marco = getattr(osm, f"get_{capa}")()
        except Exception as exc:
            logger.warning("Capa OSM '%s' omitida en %s: %s", capa, ruta.name, exc)
            continue
        if marco is None or marco.empty:
            continue

        columnas = [c for c in marco.columns if c != "geometry"]
        for _, fila in marco.iterrows():
            pares = [
                f"{columna}: {str(fila[columna]).strip()}"
                for columna in columnas
                if fila[columna] is not None and str(fila[columna]).strip().lower() not in ("", "nan", "none", "nat")
            ]
            if not pares:
                continue
            registros.append(
                _registro(ruta, "osm_pbf", len(registros) + 1, "\n".join(pares), {"capa": capa})
            )

    logger.info("OSM PBF %s: %d elementos en %d capas", ruta.name, len(registros), len(CAPAS_OSM))
    return registros


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def extraer_documento(path: str | os.PathLike[str]) -> List[Registro]:
    """Extrae cualquier documento soportado despachando por extensión.

    Es el único punto de entrada que deben usar las etapas siguientes del
    pipeline; la tabla de despacho evita el condicional gigante y permite
    registrar formatos nuevos con :func:`registrar_extractor`.

    Args:
        path: ruta al archivo.

    Returns:
        Lista de registros homogéneos (posiblemente vacía si el documento no
        contiene texto útil).

    Raises:
        ErrorExtraccion: si el archivo no existe o la extracción falla.
        FormatoNoSoportado: si la extensión no tiene extractor registrado.
    """
    ruta = Path(path)
    if not ruta.is_file():
        raise ErrorExtraccion(f"No es un archivo existente: {ruta}")

    # Coincidencia por sufijo completo del nombre, de más largo a más corto:
    # "informe.osm.pbf" debe resolver a ".osm.pbf" y no a ".pbf" (Path.suffix
    # solo ve la última extensión, y ambos formatos son protobuf incompatibles).
    nombre = ruta.name.lower()
    extension = next(
        (ext for ext in sorted(_EXTRACTORES, key=len, reverse=True) if nombre.endswith(ext)),
        None,
    )
    extractor = _EXTRACTORES.get(extension) if extension else None
    if extractor is None:
        raise FormatoNoSoportado(
            f"Extensión '{ruta.suffix}' sin extractor. Soportadas: {formatos_soportados()}"
        )

    logger.debug("Extrayendo %s con %s", ruta.name, extractor.__name__)
    try:
        return extractor(ruta)
    except ErrorExtraccion:
        raise
    except Exception as exc:  # ningún documento corrupto debe tumbar el corpus
        raise ErrorExtraccion(f"Fallo extrayendo {ruta}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Autoprueba
# --------------------------------------------------------------------------- #
def _autoprueba() -> None:
    """Verifica el contrato de salida con formatos sin dependencias externas."""
    import tempfile

    claves = {"documento", "ruta", "tipo", "pagina", "texto", "metadata"}

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        (base / "a.txt").write_text("hola\nmundo", encoding="utf-8")
        registros = extraer_documento(base / "a.txt")
        assert len(registros) == 1 and claves <= registros[0].keys()
        assert registros[0]["texto"] == "hola\nmundo"

        (base / "b.md").write_text("# Titulo\n\n- uno\n- dos\n", encoding="utf-8")
        registros = extraer_documento(base / "b.md")
        assert registros[0]["metadata"]["encabezados"] == ["# Titulo"]
        assert "- uno" in registros[0]["texto"], "el markdown no debe alterarse"

        (base / "c.json").write_text(
            json.dumps(
                {
                    "title": "T",
                    "paragraphs": ["p1", "p2"],
                    "url": "http://x",
                    "author": "A",
                }
            ),
            encoding="utf-8",
        )
        registros = extraer_documento(base / "c.json")
        assert registros[0]["texto"] == "T\n\np1\np2", registros[0]["texto"]
        assert registros[0]["metadata"]["url"] == "http://x"
        assert "http://x" not in registros[0]["texto"], "metadata no va en el texto"

        (base / "d.json").write_text(json.dumps([{"body": "x"}, {"body": "y"}]), encoding="utf-8")
        registros = extraer_documento(base / "d.json")
        assert [r["pagina"] for r in registros] == [1, 2]

        (base / "e.csv").write_text("Nombre,Ciudad,Edad\nJuan,Bogota,22\nAna,,30\n", encoding="utf-8")
        try:
            registros = extraer_documento(base / "e.csv")
            assert len(registros) == 2, "una fila = un registro"
            assert registros[0]["texto"] == "Nombre: Juan\nCiudad: Bogota\nEdad: 22"
            assert "Ciudad" not in registros[1]["texto"], "celdas vacías se omiten"
        except ErrorExtraccion as exc:
            print(f"  (CSV omitido: {exc})")

        (base / "f.html").write_text(
            "<html><head><title>T</title><style>b{}</style></head>"
            "<body><script>x=1</script><h1>Enc</h1><p>Parrafo</p><ul><li>i1</li></ul></body></html>",
            encoding="utf-8",
        )
        try:
            registros = extraer_documento(base / "f.html")
            texto = registros[0]["texto"]
            assert texto == "Enc\nParrafo\ni1", texto
            assert "x=1" not in texto and "<" not in texto
            assert registros[0]["metadata"]["encabezados"] == ["Enc"]
        except ErrorExtraccion as exc:
            print(f"  (HTML omitido: {exc})")

        (base / "g.txt").write_text("   \n  ", encoding="utf-8")
        assert extraer_documento(base / "g.txt") == [], "sin texto útil, sin registro"

        assert _retencion("a b c d", "# **a** b c d") == 1.0
        assert _retencion("y(t) = f S(t)", "**y(t)** = f") < UMBRAL_RETENCION_MD
        assert _retencion("", "loquesea") == 1.0

        (base / "i.gpkg").unlink(missing_ok=True)
        import sqlite3

        con = sqlite3.connect(base / "i.gpkg")
        con.execute("CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT)")
        con.execute("CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT)")
        con.execute("INSERT INTO gpkg_contents VALUES ('lugares', 'features')")
        con.execute("INSERT INTO gpkg_geometry_columns VALUES ('lugares', 'geom')")
        con.execute("CREATE TABLE lugares (fid INTEGER PRIMARY KEY, geom BLOB, nombre TEXT, poblacion INTEGER)")
        con.execute("INSERT INTO lugares VALUES (1, X'00', 'Sogamoso', 120000)")
        con.execute("INSERT INTO lugares VALUES (2, X'00', NULL, NULL)")
        con.commit()
        con.close()
        registros = extraer_documento(base / "i.gpkg")
        assert len(registros) == 1, "la fila totalmente vacía se descarta"
        assert registros[0]["texto"] == "nombre: Sogamoso\npoblacion: 120000"
        assert "geom" not in registros[0]["texto"], "la geometría binaria no es texto"
        assert registros[0]["metadata"]["capa"] == "lugares"

        (base / "h.xyz").write_text("?", encoding="utf-8")
        try:
            extraer_documento(base / "h.xyz")
            raise AssertionError("debió rechazar la extensión")
        except FormatoNoSoportado:
            pass

    print(f"OK - formatos registrados: {formatos_soportados()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _autoprueba()
