"""Tokenización y fragmentación: etapa 2 del pipeline RAG.

Toma los registros de :mod:`extraccion` (una página, una fila, un documento)
y los parte en *fragmentos* que quepan en la ventana del modelo de embeddings.

Contrato de salida (superconjunto del registro de entrada)::

    {
        "documento": "informe.pdf",
        "ruta": "C:/corpus/informe.pdf",
        "tipo": "pdf",
        "pagina": 1,
        "fragmento": 1,        # ordinal 1-based dentro del registro
        "texto": "...",        # <= MAX_TOKENS tokens del tokenizador
        "metadata": {..., "tokens": 487, "oraciones": 12},
    }

Reglas de diseño:

* Se corta por frontera de oración, nunca a mitad de frase: un fragmento que
  empieza en "…y por lo tanto" no recupera nada útil.
* El presupuesto se mide con el tokenizador **del modelo de embeddings**, no
  con palabras: 300 palabras de español pueden ser 600 tokens de un modelo
  multilingüe, y el modelo trunca en silencio lo que sobra.
* Los fragmentos se solapan una oración para no perder la frase que une dos
  ideas cortadas por el límite.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

__all__ = [
    "Fragmento",
    "MAX_TOKENS",
    "MAX_WORDS",
    "MODELO_TOKENIZADOR",
    "detectar_idioma",
    "inferir_fenomeno",
    "fragmentar",
    "fragmentar_registros",
    "contar_tokens",
    "contar_palabras",
]

logger = logging.getLogger(__name__)

Fragmento = Dict[str, Any]

#: Presupuesto conservador para dejar margen a tokens especiales del encoder.
#: El límite adicional de palabras facilita cumplir posteriormente el máximo de
#: 250 palabras del entregable sin tener que cortar oraciones en recuperación.
MAX_TOKENS = 500
MAX_WORDS = 220

#: Tokenizador por defecto. DEBE ser el del modelo de embeddings que se use
#: en la etapa 3; contar con otro tokenizador da un presupuesto equivocado.
MODELO_TOKENIZADOR = "intfloat/multilingual-e5-large-instruct"

#: Oraciones que se repiten al inicio del fragmento siguiente.
SOLAPE_ORACIONES = 1

#: Idioma por defecto de las reglas de segmentación de NLTK, cuando la
#: detección no alcanza a decidir (textos muy cortos o sin palabras comunes).
IDIOMA = "spanish"

#: Palabras funcionales de alta frecuencia por idioma del corpus (es/en/pt).
#: Solo hay que distinguir entre estos tres para elegir el modelo Punkt, así
#: que un conteo de stopwords basta y evita una dependencia de detección de
#: idioma. Se eligen palabras que NO se solapan entre los tres idiomas.
_MARCAS_IDIOMA = {
    "spanish": ("el", "los", "las", "del", "una", "por", "con", "para", "pero", "más", "año"),
    "english": ("the", "of", "and", "to", "is", "are", "with", "from", "this", "that", "which"),
    "portuguese": ("os", "as", "dos", "das", "uma", "com", "não", "são", "mais", "para", "pelo"),
}

_PALABRAS = re.compile(r"[^\W\d_]+", re.UNICODE)
_ENCABEZADO = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_LISTA_O_TABLA = re.compile(r"^\s*(?:[-*+]\s+|\|)")


# --------------------------------------------------------------------------- #
# Tokenizador (perezoso: se carga una vez y se reutiliza)
# --------------------------------------------------------------------------- #
_TOKENIZADORES: Dict[str, Any] = {}


def _tokenizador(nombre: str = MODELO_TOKENIZADOR) -> Any:
    """Devuelve el tokenizador de HuggingFace, cacheado por nombre.

    Raises:
        RuntimeError: si falta ``transformers`` o el modelo no se puede cargar.
    """
    if nombre not in _TOKENIZADORES:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Falta la dependencia 'transformers'. Instala: pip install transformers"
            ) from exc
        _TOKENIZADORES[nombre] = AutoTokenizer.from_pretrained(nombre)
    return _TOKENIZADORES[nombre]


def contar_tokens(texto: str, tokenizador: Any = None) -> int:
    """Cuenta tokens de contenido, sin los especiales que añade el modelo."""
    tok = tokenizador or _tokenizador()
    return len(tok.encode(texto, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# Segmentación en oraciones
# --------------------------------------------------------------------------- #
def detectar_idioma(texto: str) -> str:
    """Devuelve el idioma Punkt ('spanish', 'english' o 'portuguese') del texto.

    El corpus del reto mezcla los tres idiomas y cada uno tiene sus propias
    abreviaturas: el Punkt español corta dentro de "Dr. Smith" o "U.S. Army"
    porque no conoce esas formas, y eso parte oraciones a la mitad —
    justo lo que la especificación prohíbe (§3.3).

    Cuenta palabras funcionales exclusivas de cada idioma. No hace falta un
    detector general: solo hay que elegir entre tres segmentadores, y la
    decisión es robusta con unas pocas decenas de palabras de texto.
    """
    palabras = Counter(p.lower() for p in _PALABRAS.findall(texto))
    if not palabras:
        return IDIOMA
    puntajes = {
        idioma: sum(palabras[m] for m in marcas)
        for idioma, marcas in _MARCAS_IDIOMA.items()
    }
    mejor = max(puntajes, key=lambda k: puntajes[k])
    return mejor if puntajes[mejor] else IDIOMA


def _oraciones(texto: str, idioma: str | None = None) -> List[str]:
    """Parte un bloque en oraciones completas con NLTK Punkt."""
    import nltk

    idioma = idioma or detectar_idioma(texto)
    try:
        return nltk.sent_tokenize(texto, language=idioma)
    except LookupError:
        logger.info("Descargando el modelo Punkt de NLTK (una sola vez)")
        nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(texto, language=idioma)


def contar_palabras(texto: str) -> int:
    """Cuenta palabras lingüísticas, ignorando puntuación y números aislados."""
    return len(_PALABRAS.findall(texto))


def _secciones(texto: str) -> List[tuple[str, str, int]]:
    """Devuelve bloques estructurales ``(cuerpo, título, nivel)``."""
    secciones: List[tuple[str, str, int]] = []
    cuerpo: List[str] = []
    titulo = ""
    nivel = 0

    def cerrar() -> None:
        nonlocal cuerpo, titulo, nivel
        contenido = "\n".join(cuerpo).strip()
        if contenido or titulo:
            secciones.append((contenido, titulo, nivel))
        cuerpo = []
        titulo = ""

    for linea in texto.splitlines():
        match = _ENCABEZADO.match(linea)
        if match:
            if any(c.strip() for c in cuerpo):
                cerrar()
                titulo = match.group(2).strip()
                nivel = len(match.group(1))
            else:
                nuevo = match.group(2).strip()
                titulo = f"{titulo} - {nuevo}" if titulo else nuevo
                nivel = len(match.group(1))
                cuerpo = []  # clear any whitespace-only lines
        else:
            cuerpo.append(linea)
    cerrar()
    return secciones or [(texto.strip(), "", 0)]


def _unidades_seccion(cuerpo: str, titulo: str, nivel: int, idioma: str) -> List[tuple[str, str, int]]:
    """Convierte una sección en unidades oracionales sin cortes artificiales."""
    if not cuerpo.strip():
        return []

    unidades: List[tuple[str, str, int]] = []
    bloques = [bloque.strip() for bloque in re.split(r"\n\s*\n", cuerpo) if bloque.strip()]
    for bloque in bloques:
        lineas = bloque.splitlines()
        terminales = sum(bool(re.search(r"[.!?;:]\s*$", linea.strip())) for linea in lineas)
        cortas = sum(len(linea.strip()) <= 80 for linea in lineas)
        numericas = sum(bool(re.search(r"\d", linea)) for linea in lineas)
        total_caracteres = max(1, sum(len(linea) for linea in lineas))
        digitos = sum(caracter.isdigit() for linea in lineas for caracter in linea)
        tiene_cabecera_tabla = bool(
            re.search(r"\b(?:cuadro|tabla|table|figure|figura)\s+\d+\b", " ".join(lineas), re.IGNORECASE)
        )
        bloque_denso = (
            len(lineas) >= 4
            and (
                tiene_cabecera_tabla
                or (
                    terminales / len(lineas) < 0.5
                    and (
                        cortas / len(lineas) >= 0.6
                        or numericas / len(lineas) >= 0.4
                        or digitos / total_caracteres >= 0.08
                    )
                )
            )
        )
        if bloque_denso:
            # Las tablas extraídas de PDF suelen perder delimitadores y llegar
            # como filas cortas sin puntuación. Cada línea es aquí la unidad
            # atómica; se empaquetan varias filas, pero nunca se parten.
            unidades.extend(
                (linea.strip(), titulo, nivel)
                for linea in lineas
                if linea.strip()
            )
            continue

        partes: List[str] = []
        for linea in lineas:
            if len(lineas) > 1 and _LISTA_O_TABLA.match(linea):
                if partes:
                    unidades.extend((u.strip(), titulo, nivel) for u in _oraciones(" ".join(partes), idioma) if u.strip())
                    partes = []
                unidades.extend((u.strip(), titulo, nivel) for u in _oraciones(linea, idioma) if u.strip())
            else:
                partes.append(linea)
        if partes:
            unidades.extend((u.strip(), titulo, nivel) for u in _oraciones("\n".join(partes), idioma) if u.strip())

    if titulo and unidades:
        primera, _, _ = unidades[0]
        unidades[0] = (f"{titulo}: {primera}", titulo, nivel)
    return unidades


# --------------------------------------------------------------------------- #
# Empaquetado
# --------------------------------------------------------------------------- #
def _empacar(
    oraciones: Sequence[str],
    costos: Sequence[int],
    limite: int,
    solape: int,
    costos_palabras: Sequence[int] | None = None,
    limite_palabras: int | None = None,
) -> List[List[int]]:
    """Agrupa unidades sin superar los límites de tokens y palabras.

    Las unidades se tratan como atómicas: nunca se dividen por tokens. Una
    oración o fila individual que exceda el presupuesto se conserva completa y
    queda marcada como ``oversize`` por la etapa superior.
    """
    grupos: List[List[int]] = []
    actual: List[int] = []
    total_tokens = 0
    total_palabras = 0
    costos_palabras = costos_palabras or [0] * len(costos)

    for indice, costo in enumerate(costos):
        palabras = costos_palabras[indice]
        excede_tokens = bool(actual and total_tokens + costo > limite)
        excede_palabras = bool(
            actual and limite_palabras is not None
            and total_palabras + palabras > limite_palabras
        )
        if excede_tokens or excede_palabras:
            grupos.append(actual)
            actual = actual[-solape:] if solape else []
            total_tokens = sum(costos[i] for i in actual)
            total_palabras = sum(costos_palabras[i] for i in actual)
            if (
                total_tokens + costo > limite
                or (limite_palabras is not None and total_palabras + palabras > limite_palabras)
            ):
                actual = []
                total_tokens = total_palabras = 0
        actual.append(indice)
        total_tokens += costo
        total_palabras += palabras

    if actual:
        grupos.append(actual)
    return grupos


def _ajustar_grupo(
    textos: Sequence[str],
    grupo: Sequence[int],
    limite_tokens: int,
    limite_palabras: int,
    tokenizador: Any,
) -> List[List[int]]:
    """Revalida grupos con el texto concatenado, no solo costes individuales."""
    salida: List[List[int]] = []
    actual: List[int] = []
    for indice in grupo:
        candidato = actual + [indice]
        texto = " ".join(textos[i] for i in candidato).strip()
        excede = (
            contar_tokens(texto, tokenizador) > limite_tokens
            or contar_palabras(texto) > limite_palabras
        )
        if actual and excede:
            salida.append(actual)
            actual = [indice]
        else:
            actual = candidato
    if actual:
        salida.append(actual)
    return salida


def _fragmentar_detallado(
    texto: str,
    max_tokens: int,
    max_words: int,
    solape: int,
    tokenizador: Any,
    idioma: str | None,
) -> List[tuple[str, int, int, str, int, bool]]:
    """Fragmenta y devuelve trazabilidad estructural y de tamaño."""
    idioma = idioma or detectar_idioma(texto)
    unidades: List[tuple[str, str, int]] = []
    for cuerpo, titulo, nivel in _secciones(texto):
        unidades.extend(_unidades_seccion(cuerpo, titulo, nivel, idioma))

    salida: List[tuple[str, int, int, str, int, bool]] = []
    cursor = 0
    while cursor < len(unidades):
        titulo = unidades[cursor][1]
        nivel = unidades[cursor][2]
        fin = cursor
        while fin < len(unidades) and unidades[fin][1] == titulo and unidades[fin][2] == nivel:
            fin += 1

        bloque = unidades[cursor:fin]
        textos = [unidad[0] for unidad in bloque]
        costos = [contar_tokens(unidad, tokenizador) for unidad in textos]
        palabras = [contar_palabras(unidad) for unidad in textos]
        grupos = _empacar(textos, costos, max_tokens, solape, palabras, max_words)
        for grupo in grupos:
            for subgrupo in _ajustar_grupo(textos, grupo, max_tokens, max_words, tokenizador):
                fragmento = " ".join(textos[i] for i in subgrupo).strip()
                if not fragmento:
                    continue
                num_tokens = contar_tokens(fragmento, tokenizador)
                num_palabras = contar_palabras(fragmento)
                oversize = num_tokens > max_tokens or num_palabras > max_words
                salida.append((fragmento, num_tokens, num_palabras, titulo, nivel, oversize))
        cursor = fin
    return salida


def fragmentar(
    texto: str,
    max_tokens: int = MAX_TOKENS,
    solape: int = SOLAPE_ORACIONES,
    tokenizador: Any = None,
    idioma: str | None = None,
    max_words: int = MAX_WORDS,
) -> List[tuple[str, int]]:
    """Parte texto en unidades completas con límites de tokens y palabras."""
    tok = tokenizador or _tokenizador()
    return [
        (fragmento, tokens)
        for fragmento, tokens, _, _, _, _ in _fragmentar_detallado(
            texto, max_tokens, max_words, solape, tok, idioma
        )
    ]


KEYWORDS_FENOMENO: Dict[int, List[str]] = {
    1: ["inteligencia artificial", "ia", "militar", "defensa", "armas", "autónom", "autonom", "warfare", "military", "defense", "weapon", "combate", "fuerzas armadas"],
    2: ["espacial", "espacio", "órbita", "orbita", "leo", "debris", "basura espacial", "satélite", "satelite", "orbital", "space", "satellite", "astronom", "cosmo"],
    3: ["américa latina", "latina", "caribe", "territorial", "migración", "migracion", "gobernanza", "violencia", "colombia", "latinoamérica", "latinoamerica", "región", "region", "conflicto", "social"],
}


def inferir_fenomeno(ruta: Path | str, texto: str = "") -> int:
    """Infiere el fenómeno temático (1, 2 o 3) garantizando nunca retornar None.

    1. Busca 'fenomeno_1', 'fenomeno_2', 'fenomeno_3' en la ruta.
    2. Si no lo encuentra, analiza coincidencia de palabras clave en la ruta y el texto.
    3. Si hay empate o 0 coincidencias, por defecto asigna 1.
    """
    path_obj = Path(ruta)
    for parte in path_obj.parts:
        parte_lower = parte.lower()
        if "fenomeno_1" in parte_lower or "fenomeno1" in parte_lower or parte_lower == "f1":
            return 1
        if "fenomeno_2" in parte_lower or "fenomeno2" in parte_lower or parte_lower == "f2":
            return 2
        if "fenomeno_3" in parte_lower or "fenomeno3" in parte_lower or parte_lower == "f3":
            return 3

    contenido = f"{path_obj.name} {texto[:5000]}".lower()
    puntajes = {
        fen: sum(contenido.count(kw) for kw in keywords)
        for fen, keywords in KEYWORDS_FENOMENO.items()
    }
    max_score = max(puntajes.values())
    if max_score > 0:
        for fen, score in puntajes.items():
            if score == max_score:
                return fen
    return 1


def fragmentar_registros(
    registros: Iterable[Dict[str, Any]],
    max_tokens: int = MAX_TOKENS,
    solape: int = SOLAPE_ORACIONES,
    tokenizador: Any = None,
    max_words: int = MAX_WORDS,
) -> List[Fragmento]:
    """Aplica :func:`fragmentar` a los registros de :mod:`extraccion`.

    La salida sigue el esquema estándar del pipeline (alineado con chunks.jsonl):

    .. code-block:: json

        {
            "doc_id":    "DOC-0001",
            "chunk_id":  "DOC-0001-chunk-000",
            "fuente":    "fenomeno_1/archivo.pdf",
            "formato":   "pdf",
            "fenomeno":  1,
            "posicion":  0,
            "num_tokens": 487,
            "texto":     "..."
        }

    ``posicion`` es 0-based y continuo por documento (no por página).
    El idioma detectado y la metadata de procedencia se conservan en ``_meta``.
    """
    from pathlib import Path
    from extraccion.extraccion import _normalizar_formato, _obtener_fuente_relativa

    tok = tokenizador or _tokenizador()

    fragmentos: List[Fragmento] = []
    posiciones_por_doc: Dict[str, int] = {}

    for registro in registros:
        doc_id: str = registro.get("doc_id", "DOC-0000")
        documento: str = registro.get("documento", "")
        ruta_str: str = registro.get("ruta", "")

        # 1.2: Sanear el campo fuente (ruta relativa estandarizada)
        fuente: str = registro.get("fuente") or (
            _obtener_fuente_relativa(Path(ruta_str)) if ruta_str else documento
        )

        # 1.3: Normalizar el campo formato a los canónicos
        tipo_raw: str = registro.get("tipo", "")
        formato: str = _normalizar_formato(tipo_raw, Path(fuente))

        # 1.4: Inferencia y asignación garantizada del campo fenomeno (nunca None)
        fenomeno_raw = registro.get("metadata", {}).get("fenomeno")
        if fenomeno_raw in (1, 2, 3):
            fenomeno = int(fenomeno_raw)
        else:
            fenomeno = inferir_fenomeno(fuente or ruta_str, registro.get("texto", ""))

        idioma = detectar_idioma(registro["texto"])
        trozos = _fragmentar_detallado(
            registro["texto"], max_tokens, max_words, solape, tok, idioma
        )

        # 1.1: Contador continuo de posicion y chunk_id por doc_id
        pos_inicio = posiciones_por_doc.get(doc_id, 0)
        for i, (texto, tokens, palabras, seccion, nivel_seccion, oversize) in enumerate(trozos):
            posicion = pos_inicio + i
            fragmentos.append(
                {
                    "doc_id":    doc_id,
                    "chunk_id":  f"{doc_id}-chunk-{posicion:03d}",
                    "fuente":    fuente,
                    "formato":   formato,
                    "fenomeno":  fenomeno,
                    "posicion":  posicion,
                    "num_tokens": tokens,
                    "texto":     texto,
                    # Metadata auxiliar de trazabilidad
                    "_meta": {
                        **registro.get("metadata", {}),
                        "idioma": idioma,
                        "pagina": registro.get("pagina"),
                        "fragmentos_pagina": len(trozos),
                        "seccion": seccion or None,
                        "nivel_seccion": nivel_seccion or None,
                        "num_palabras": palabras,
                        "oversize": oversize,
                    },
                }
            )
        posiciones_por_doc[doc_id] = pos_inicio + len(trozos)

    logger.info("Fragmentación: %d fragmentos", len(fragmentos))
    return fragmentos


# --------------------------------------------------------------------------- #
# Autoprueba
# --------------------------------------------------------------------------- #
def _autoprueba() -> None:
    """Verifica el empaquetado sin red y el pipeline completo si hay modelo."""
    # _empacar es la lógica de verdad y no depende del tokenizador.
    assert _empacar(["a", "b", "c"], [4, 4, 4], 10, 0) == [[0, 1], [2]]
    assert _empacar(["a", "b", "c"], [4, 4, 4], 10, 1) == [[0, 1], [1, 2]]
    assert _empacar(["a"], [99], 10, 1) == [[0]], "una oración sola siempre sale"
    assert _empacar([], [], 10, 1) == []
    # el solape no debe desbordar el presupuesto: dos oraciones de 5 con límite 5
    assert _empacar(["a", "b"], [5, 5], 5, 1) == [[0], [1]]

    # Ningún grupo puede exceder el presupuesto (salvo oración única indivisible).
    costos = [7, 3, 9, 1, 5, 5, 2]
    for grupo in _empacar(list("abcdefg"), costos, 12, 1):
        assert len(grupo) == 1 or sum(costos[i] for i in grupo) <= 12, grupo

    # La detección solo tiene que acertar entre los tres idiomas del corpus.
    assert detectar_idioma(
        "El informe de la agencia describe los riesgos del entorno orbital "
        "con más detalle que el año pasado, pero sin datos nuevos."
    ) == "spanish"
    assert detectar_idioma(
        "The report of the agency describes the risks of the orbital "
        "environment and the debris that is tracked from the ground."
    ) == "english"
    assert detectar_idioma(
        "O relatório da agência descreve os riscos das órbitas com uma "
        "análise dos detritos, não são dados novos, mais informação pelo site."
    ) == "portuguese"
    assert detectar_idioma("") == IDIOMA, "sin palabras cae al idioma por defecto"
    assert detectar_idioma("123 456 ...") == IDIOMA

    try:
        tok = _tokenizador()
    except Exception as exc:  # sin red o sin transformers: la parte offline ya pasó
        print(f"OK (solo empaquetado) - tokenizador no disponible: {exc}")
        return

    texto = "Fig. 1 muestra el caso. " + "Una oración de prueba. " * 200
    trozos = fragmentar(texto, max_tokens=100, tokenizador=tok)
    assert len(trozos) > 1, "debió partirse en varios fragmentos"
    assert all(t <= 100 for _, t in trozos), [t for _, t in trozos]
    assert "Fig. 1 muestra el caso." in trozos[0][0], "Punkt no debe cortar en 'Fig.'"

    largo = "Primera oración completa. " + "Segunda oración completa. " * 200
    largos = fragmentar(largo, max_tokens=50, max_words=40, tokenizador=tok)
    assert len(largos) > 1
    assert all("oración" in texto for texto, _ in largos)

    registro = {
        "doc_id": "DOC-0001", "documento": "d.pdf", "ruta": "/d.pdf", "tipo": "pdf", "pagina": 3,
        "texto": texto, "metadata": {"origen_texto": "nativo"},
    }
    fragmentos = fragmentar_registros([registro], max_tokens=100, tokenizador=tok)
    assert [f["posicion"] for f in fragmentos] == list(range(len(fragmentos)))
    assert all(f["_meta"]["pagina"] == 3 and f["fuente"] == "d.pdf" for f in fragmentos)
    assert fragmentos[0]["_meta"]["origen_texto"] == "nativo", "metadata heredada"

    # Prueba multi-página: posicion debe ser continua (0, 1, 2...) y chunk_id único
    reg_p1 = {**registro, "pagina": 1, "texto": "Primera página del documento."}
    reg_p2 = {**registro, "pagina": 2, "texto": "Segunda página del documento."}
    frag_multi = fragmentar_registros([reg_p1, reg_p2], max_tokens=100, tokenizador=tok)
    assert [f["posicion"] for f in frag_multi] == [0, 1]
    assert [f["chunk_id"] for f in frag_multi] == ["DOC-0001-chunk-000", "DOC-0001-chunk-001"]

    assert fragmentar_registros([{**registro, "texto": "   "}], tokenizador=tok) == []

    print(f"OK - {len(fragmentos)} fragmentos, max {max(f['num_tokens'] for f in fragmentos)} tokens")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _autoprueba()
