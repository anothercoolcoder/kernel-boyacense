"""Audita metadata y resultados contra inventario oficial."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from lib.inventario_oficial import cargar_inventario, resolver_archivo

FORMATOS = {"pdf", "json", "csv", "xlsx", "jpg", "avif", "txt", "pbf"}
EXTENSIONES = tuple(f".{formato}" for formato in FORMATOS)


def cargar_jsonl(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def auditar(
    metadata_path: Path,
    inventario_path: Path,
    resultados_path: Path | None,
    corpus_path: Path | None,
) -> None:
    inventario = cargar_inventario(inventario_path)
    if corpus_path:
        oficiales = {
            resolver_archivo(archivo, inventario, inventario_path.parent).get("doc_id")
            for archivo in corpus_path.rglob("*")
            if archivo.is_file()
            and any(archivo.name.lower().endswith(ext) for ext in EXTENSIONES)
        }
    else:
        oficiales = {registro["DOC_ID"] for registro in inventario}
    metadata = cargar_jsonl(metadata_path)
    errores: list[str] = []
    ids = {registro.get("doc_id") for registro in metadata}
    chunk_ids = [str(registro.get("chunk_id")) for registro in metadata]
    if len(set(chunk_ids)) != len(chunk_ids):
        errores.append("chunk_id duplicado en metadata")
    for registro in metadata:
        doc_id = str(registro.get("doc_id", ""))
        chunk_id = str(registro.get("chunk_id", ""))
        if not re.fullmatch(rf"{re.escape(doc_id)}-chunk-\d+", chunk_id):
            errores.append(f"chunk_id invalido: {chunk_id}")
    index_path = metadata_path.parent / "index.faiss"
    if index_path.is_file():
        try:
            import faiss

            if faiss.read_index(str(index_path)).ntotal != len(metadata):
                errores.append("cantidad de vectores FAISS no coincide con metadata")
        except ImportError:
            errores.append("no se pudo importar faiss para auditar index.faiss")
    faltantes = oficiales - ids
    if faltantes:
        errores.append(f"documentos sin metadata: {len(faltantes)}")
    inventario_por_id = {registro["DOC_ID"]: registro for registro in inventario}
    for indice, registro in enumerate(metadata):
        doc_id = registro.get("doc_id")
        oficial = inventario_por_id.get(doc_id)
        if oficial is None:
            errores.append(f"doc_id no oficial: {doc_id}")
            continue
        esperado = {"F1": 1, "F2": 2, "F3": 3}[oficial["Fenómeno"]]
        if registro.get("fenomeno") != esperado:
            errores.append(f"fenomeno incorrecto: {doc_id}")
        if registro.get("formato") not in FORMATOS:
            errores.append(f"formato invalido: {doc_id}={registro.get('formato')}")
        fuente = f"{oficial['Carpeta'].strip('/')}/{oficial['Nombre estandarizado']}"
        if registro.get("fuente") != fuente:
            errores.append(f"fuente incorrecta: {doc_id}")
    por_doc: dict[str, list[int]] = {}
    for registro in metadata:
        por_doc.setdefault(registro.get("doc_id", ""), []).append(registro.get("posicion", -1))
    for doc_id, posiciones in por_doc.items():
        if sorted(posiciones) != list(range(len(posiciones))):
            errores.append(f"posicion no continua desde cero: {doc_id}")
    if resultados_path:
        resultados = cargar_jsonl(resultados_path)
        for resultado in resultados:
            doc_ids = [item.get("doc_id") for item in resultado.get("documents", [])]
            chunk_ids = [item.get("chunk_id") for item in resultado.get("fragments", [])]
            if any(doc_id not in oficiales for doc_id in doc_ids):
                errores.append(f"resultado con doc_id no oficial: {resultado.get('query_id')}")
            if len(doc_ids) != 3 or len(chunk_ids) != 10:
                errores.append(f"cardinalidad invalida: {resultado.get('query_id')}")
            if len(set(doc_ids)) != len(doc_ids) or len(set(chunk_ids)) != len(chunk_ids):
                errores.append(f"IDs duplicados: {resultado.get('query_id')}")
            if any(len(item.get("text", "").split()) > 250 for item in resultado.get("fragments", [])):
                errores.append(f"fragmento largo: {resultado.get('query_id')}")
    resumen = Counter(registro.get("fenomeno") for registro in metadata)
    print(json.dumps({"metadata": len(metadata), "documentos": len(ids), "fenomenos": resumen, "errores": errores}, ensure_ascii=False, default=dict))
    if errores:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--inventario", type=Path, default=Path("CORPUS CODEFEST AD ASTRA 2026/Indice_Datos_Codefest.xlsx"))
    parser.add_argument("--resultados", type=Path)
    parser.add_argument("--corpus", type=Path, help="Alcance de cobertura; omite corpus completo")
    args = parser.parse_args()
    auditar(args.metadata, args.inventario, args.resultados, args.corpus)


if __name__ == "__main__":
    main()
