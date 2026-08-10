"""Prepara borrador de ground truth desde preguntas, respuestas y metadata."""

from __future__ import annotations

import json
import re
import argparse
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
METADATA = ROOT / "base_vectorial/encoder_multilingual-e5-large-instruct-prueba/metadata.jsonl"
PREGUNTAS = ROOT / "preguntas.txt"
RESPUESTAS = ROOT / "respuestas.txt"
SALIDA = ROOT / "context/benchmark_rag_oficial_prueba.jsonl"

SECCIONES = {
    "ILLIA": ["F1-ILIA-005"],
    "INPE": ["F2-INPE-057", "F2-INPE-058", "F2-INPE-059"],
    "SWF_inve": ["F2-SWF-128"],
    "SIPRI_NUPI_FACT": ["F3-SIPRI-107"],
    "SIPRI_RRP": ["F3-SIPRI-122"],
    "MAPPOEA_cp-informe-mappoea-39-esp": ["F3-MAPPOEA-014"],
    "MAPPOEA_desafios-de-gestion-del-conocimiento-para-la-construccion-de-paz-af": ["F3-MAPPOEA-017"],
    "ESA_design-for-demise-verification-guidelines-v1": ["F2-ESA-013"],
    "AIINDEX_ai-index-report-2026": ["F1-AIINDEX-023"],
}


def normalizar(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    return re.findall(r"\w+", texto)


def cargar_preguntas() -> list[str]:
    return [
        linea.strip()
        for linea in PREGUNTAS.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.startswith("#")
    ]


def cargar_respuestas() -> list[dict[str, str]]:
    registros: list[dict[str, str]] = []
    seccion = ""
    lineas = [linea.strip() for linea in RESPUESTAS.read_text(encoding="utf-8").splitlines()]
    for indice, linea in enumerate(lineas):
        if linea.startswith("#"):
            coincidencia = re.search(r"DOC\s+(.+)$", linea)
            if coincidencia:
                seccion = coincidencia.group(1).strip()
            continue
        es_pregunta = linea.endswith("?")
        if not es_pregunta:
            continue
        if indice + 1 >= len(lineas) or not lineas[indice + 1]:
            raise ValueError(f"Respuesta ausente para: {linea}")
        registros.append({"query": linea, "answer": lineas[indice + 1], "source_group": seccion})
    return registros


def cargar_metadata(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def candidatos(answer: str, documentos: set[str], metadata: list[dict]) -> list[dict]:
    tokens_respuesta = set(token for token in normalizar(answer) if len(token) > 2)
    por_documento: dict[str, list[dict]] = defaultdict(list)
    for meta in metadata:
        if meta.get("doc_id") not in documentos:
            continue
        tokens_chunk = set(normalizar(meta.get("texto", "")))
        coincidencias = tokens_respuesta & tokens_chunk
        score = sum(2 if token.isdigit() else 1 for token in coincidencias)
        if score:
            por_documento[meta["doc_id"]].append({
                "chunk_id": meta.get("chunk_id"),
                "doc_id": meta.get("doc_id"),
                "score": score,
                "pagina": meta.get("pagina"),
                "section": meta.get("seccion", ""),
            })
    resultado = [item for items in por_documento.values() for item in items]
    resultado.sort(key=lambda item: (-item["score"], item["doc_id"], item["chunk_id"]))
    return resultado[:5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()
    preguntas = cargar_preguntas()
    respuestas = cargar_respuestas()
    if len(preguntas) != len(respuestas):
        raise ValueError(f"Preguntas ({len(preguntas)}) y respuestas ({len(respuestas)}) no alinean")

    metadata = cargar_metadata(args.metadata)
    args.salida.parent.mkdir(exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as archivo:
        for indice, (pregunta, respuesta) in enumerate(zip(preguntas, respuestas), 1):
            documentos = set(SECCIONES.get(respuesta["source_group"], []))
            candidatos_encontrados = candidatos(respuesta["answer"], documentos, metadata)
            registro = {
                "query_id": f"q{indice:03d}",
                "intent_id": f"intent-{indice:03d}",
                "language": "en" if pregunta.startswith(("What", "How", "By ")) else "es",
                "query": pregunta,
                "answer": respuesta["answer"],
                "source_group": respuesta["source_group"],
                "expected_source_doc_ids": sorted(documentos),
                "candidate_chunks": candidatos_encontrados,
                "annotation_status": "draft",
                "annotator": None,
                "relevant_doc_ids": [],
                "relevant_chunk_ids": [],
                "graded_relevance": {},
            }
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"Borrador escrito: {args.salida}")
    print(f"Consultas: {len(preguntas)}")


if __name__ == "__main__":
    main()
