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
from typing import Any, Dict, Iterable, List, Sequence

__all__ = [
    "Fragmento",
    "MAX_TOKENS",
    "MODELO_TOKENIZADOR",
    "detectar_idioma",
    "fragmentar",
    "fragmentar_registros",
    "contar_tokens",
]

logger = logging.getLogger(__name__)

Fragmento = Dict[str, Any]

#: Tokens útiles por fragmento: 512 de ventana menos [CLS] y [SEP].
MAX_TOKENS = 510

#: Tokenizador por defecto. DEBE ser el del modelo de embeddings que se use
#: en la etapa 3; contar con otro tokenizador da un presupuesto equivocado.
MODELO_TOKENIZADOR = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
    """Parte el texto en oraciones con NLTK (Punkt), descargando el modelo si falta.

    Punkt entiende que "Fig. 1" o "et al." no terminan una oración, que es
    justo donde un ``split('.')`` destroza el texto académico.
    """
    import nltk

    idioma = idioma or detectar_idioma(texto)
    try:
        return nltk.sent_tokenize(texto, language=idioma)
    except LookupError:
        logger.info("Descargando el modelo Punkt de NLTK (una sola vez)")
        nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(texto, language=idioma)


def _trozos_duros(oracion: str, limite: int, tokenizador: Any) -> List[str]:
    """Parte por tokens una oración que por sí sola excede el límite.

    Caso real: tablas serializadas y bloques de código, que no tienen puntos
    donde cortar. Aquí sí se corta a mitad de frase porque no hay alternativa:
    dejarla pasar entera haría que el modelo truncara el resto en silencio.
    """
    ids = tokenizador.encode(oracion, add_special_tokens=False)
    trozos: List[str] = []
    inicio = 0
    while inicio < len(ids):
        fin = min(inicio + limite, len(ids))
        trozo = tokenizador.decode(ids[inicio:fin], skip_special_tokens=True).strip()
        # decode->encode no es idempotente (SentencePiece reañade prefijos ▁), así
        # que el trozo puede volver a medir más del límite: se recorta hasta caber.
        while fin > inicio + 1 and contar_tokens(trozo, tokenizador) > limite:
            fin -= 1
            trozo = tokenizador.decode(ids[inicio:fin], skip_special_tokens=True).strip()
        if trozo:
            trozos.append(trozo)
        inicio = fin
    return trozos


# --------------------------------------------------------------------------- #
# Empaquetado
# --------------------------------------------------------------------------- #
def _empacar(
    oraciones: Sequence[str],
    costos: Sequence[int],
    limite: int,
    solape: int,
) -> List[List[int]]:
    """Agrupa índices de oración en fragmentos que no superen ``limite`` tokens.

    Suma el costo por oración en vez de re-tokenizar el fragmento en cada paso
    (que sería cuadrático). La suma sobreestima ligeramente el total real
    —cada oración paga sus tokens de borde—, así que el presupuesto queda del
    lado seguro: nunca produce un fragmento que el modelo tenga que truncar.
    """
    grupos: List[List[int]] = []
    actual: List[int] = []
    total = 0

    for indice, costo in enumerate(costos):
        if actual and total + costo > limite:
            grupos.append(actual)
            actual = actual[-solape:] if solape else []
            total = sum(costos[i] for i in actual)
            if total + costo > limite:
                # el solape no deja sitio para la oración: el contexto extra se
                # sacrifica antes que desbordar el presupuesto del modelo.
                actual, total = [], 0
        actual.append(indice)
        total += costo

    if actual:
        grupos.append(actual)
    return grupos


def fragmentar(
    texto: str,
    max_tokens: int = MAX_TOKENS,
    solape: int = SOLAPE_ORACIONES,
    tokenizador: Any = None,
    idioma: str | None = None,
) -> List[tuple[str, int]]:
    """Parte un texto en fragmentos de ``<= max_tokens``, cortando por oración.

    Args:
        texto: texto a fragmentar.
        max_tokens: presupuesto de tokens de contenido por fragmento.
        solape: oraciones repetidas entre fragmentos consecutivos.
        tokenizador: tokenizador de HuggingFace; por defecto el de
            :data:`MODELO_TOKENIZADOR`.
        idioma: idioma Punkt; por defecto se detecta con
            :func:`detectar_idioma`.

    Returns:
        Lista de ``(texto_fragmento, tokens)`` en orden de lectura.
    """
    tok = tokenizador or _tokenizador()

    oraciones: List[str] = []
    for oracion in _oraciones(texto, idioma):
        oracion = oracion.strip()
        if not oracion:
            continue
        if contar_tokens(oracion, tok) > max_tokens:
            oraciones.extend(_trozos_duros(oracion, max_tokens, tok))
        else:
            oraciones.append(oracion)

    if not oraciones:
        return []

    costos = [contar_tokens(o, tok) for o in oraciones]
    salida: List[tuple[str, int]] = []
    for grupo in _empacar(oraciones, costos, max_tokens, solape):
        fragmento = " ".join(oraciones[i] for i in grupo).strip()
        if fragmento:
            salida.append((fragmento, contar_tokens(fragmento, tok)))
    return salida


def fragmentar_registros(
    registros: Iterable[Dict[str, Any]],
    max_tokens: int = MAX_TOKENS,
    solape: int = SOLAPE_ORACIONES,
    tokenizador: Any = None,
) -> List[Fragmento]:
    """Aplica :func:`fragmentar` a los registros de :mod:`extraccion`.

    Conserva la procedencia (documento, página) de cada fragmento: sin ella el
    sistema puede recuperar el texto correcto pero no decir de dónde salió.
    El idioma se detecta una vez por registro y queda marcado en la metadata,
    que es lo que pide la especificación en §2.2.
    """
    tok = tokenizador or _tokenizador()

    fragmentos: List[Fragmento] = []
    for registro in registros:
        idioma = detectar_idioma(registro["texto"])
        trozos = fragmentar(registro["texto"], max_tokens, solape, tok, idioma)
        for orden, (texto, tokens) in enumerate(trozos, start=1):
            fragmentos.append(
                {
                    **registro,
                    "fragmento": orden,
                    "texto": texto,
                    "metadata": {
                        **registro["metadata"],
                        "tokens": tokens,
                        "idioma": idioma,
                        "fragmentos_pagina": len(trozos),
                    },
                }
            )

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

    largo = "palabra " * 2000
    assert all(t <= 50 for _, t in fragmentar(largo, max_tokens=50, tokenizador=tok))

    registro = {
        "documento": "d.pdf", "ruta": "/d.pdf", "tipo": "pdf", "pagina": 3,
        "texto": texto, "metadata": {"origen_texto": "nativo"},
    }
    fragmentos = fragmentar_registros([registro], max_tokens=100, tokenizador=tok)
    assert [f["fragmento"] for f in fragmentos] == list(range(1, len(fragmentos) + 1))
    assert all(f["pagina"] == 3 and f["documento"] == "d.pdf" for f in fragmentos)
    assert fragmentos[0]["metadata"]["origen_texto"] == "nativo", "metadata heredada"

    assert fragmentar_registros([{**registro, "texto": "   "}], tokenizador=tok) == []

    print(f"OK - {len(fragmentos)} fragmentos, max {max(f['metadata']['tokens'] for f in fragmentos)} tokens")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _autoprueba()
