"""Migra ground truth aprobado entre índices con evidencia textual idéntica."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DOC_MAP = {
    "DOC-0001": "F1-AIINDEX-023",
    "DOC-0002": "F2-ESA-013",
    "DOC-0003": "F1-ILIA-005",
    "DOC-0004": "F2-INPE-057",
    "DOC-0005": "F2-INPE-058",
    "DOC-0006": "F2-INPE-059",
    "DOC-0007": "F3-MAPPOEA-014",
    "DOC-0008": "F3-MAPPOEA-017",
    "DOC-0009": "F3-SIPRI-107",
    "DOC-0010": "F3-SIPRI-122",
    "DOC-0011": "F2-SWF-128",
}


def leer_jsonl(ruta: Path) -> list[dict]:
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]


def construir_mapa(old_metadata: list[dict], new_metadata: list[dict]) -> dict[str, str]:
    mapa: dict[str, str] = {}
    for antiguo in old_metadata:
        doc_nuevo = DOC_MAP.get(antiguo.get("doc_id"))
        candidatos = [
            nuevo for nuevo in new_metadata
            if nuevo.get("doc_id") == doc_nuevo
            and nuevo.get("texto") == antiguo.get("texto")
            and nuevo.get("pagina") == antiguo.get("pagina")
        ]
        if len(candidatos) != 1:
            raise ValueError(
                f"No hay mapeo textual unico para {antiguo.get('chunk_id')}: {len(candidatos)}"
            )
        mapa[antiguo["chunk_id"]] = candidatos[0]["chunk_id"]
    return mapa


def migrar(origen: Path, old_metadata_path: Path, new_metadata_path: Path, salida: Path) -> None:
    old_gt = leer_jsonl(origen)
    old_metadata = leer_jsonl(old_metadata_path)
    new_metadata = leer_jsonl(new_metadata_path)
    mapa_chunks = construir_mapa(old_metadata, new_metadata)
    new_chunks = {item["chunk_id"] for item in new_metadata}
    salida_registros: list[dict] = []
    for registro in old_gt:
        if registro.get("annotation_status") != "approved":
            raise ValueError(f"Ground truth origen no aprobado: {registro['query_id']}")
        grados = {
            mapa_chunks[chunk_id]: grado
            for chunk_id, grado in registro.get("graded_relevance", {}).items()
        }
        relevantes = [mapa_chunks[chunk_id] for chunk_id in registro.get("relevant_chunk_ids", [])]
        if any(chunk_id not in new_chunks for chunk_id in grados) or any(
            chunk_id not in new_chunks for chunk_id in relevantes
        ):
            raise ValueError(f"Chunk migrado ausente: {registro['query_id']}")
        actualizado = dict(registro)
        actualizado["expected_source_doc_ids"] = [DOC_MAP[doc_id] for doc_id in registro["expected_source_doc_ids"]]
        actualizado["relevant_doc_ids"] = [DOC_MAP[doc_id] for doc_id in registro["relevant_doc_ids"]]
        actualizado["relevant_chunk_ids"] = relevantes
        actualizado["graded_relevance"] = grados
        actualizado["annotation_status"] = "approved"
        salida_registros.append(actualizado)
    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = salida.with_suffix(salida.suffix + ".tmp")
    temporal.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in salida_registros),
        encoding="utf-8",
    )
    temporal.replace(salida)
    print(f"Migrados: {len(salida_registros)}")
    print(f"Chunks mapeados: {len(mapa_chunks)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origen", type=Path, default=Path("context/benchmark_rag.jsonl"))
    parser.add_argument("--metadata-antigua", type=Path, default=Path("base_vectorial/encoder_multilingual-e5-large-instruct/metadata.jsonl"))
    parser.add_argument("--metadata-nueva", type=Path, default=Path("base_vectorial/encoder_multilingual-e5-large-instruct-prueba/metadata.jsonl"))
    parser.add_argument("--salida", type=Path, default=Path("context/benchmark_rag_oficial_prueba.jsonl"))
    args = parser.parse_args()
    migrar(args.origen, args.metadata_antigua, args.metadata_nueva, args.salida)


if __name__ == "__main__":
    main()
