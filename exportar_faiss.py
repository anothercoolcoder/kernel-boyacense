#!/usr/bin/env python3
"""Script para extraer, parsear y exportar datos y metadatos de un índice FAISS.

Soporta exportación a formatos JSON, JSONL, CSV y Markdown, además de resumen
estadístico y búsqueda vectorial desde CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Rutas por defecto
FAISS_DIR_DEFAULT = Path(__file__).parent / "faiss_index"
OUTPUT_DIR_DEFAULT = Path(__file__).parent / "salida"


def cargar_datos_faiss(faiss_dir: Path | str = FAISS_DIR_DEFAULT) -> List[Dict[str, Any]]:
    """Lee el archivo index.pkl de FAISS y devuelve la lista estructurada de chunks.

    Args:
        faiss_dir: Directorio donde se encuentran index.pkl e index.faiss.

    Returns:
        Lista de diccionarios con id_faiss, doc_id, chunk_id, texto, num_tokens y metadatos completos.
    """
    faiss_dir = Path(faiss_dir)
    pkl_path = faiss_dir / "index.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de metadatos en: {pkl_path}")

    with open(pkl_path, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # Invertir o mapear index_to_docstore_id
    id_map = {docstore_id: faiss_id for faiss_id, docstore_id in index_to_docstore_id.items()}

    fragmentos = []
    for docstore_id, doc in docstore._dict.items():
        faiss_id = id_map.get(docstore_id, -1)
        meta = dict(doc.metadata) if hasattr(doc, "metadata") and doc.metadata else {}

        # Normalizar campos clave para análisis estandarizado
        doc_id = meta.get("doc_id") or meta.get("documento") or meta.get("fuente") or "desconocido"
        formato = meta.get("formato") or meta.get("tipo") or meta.get("extension") or "desconocido"
        fuente = meta.get("fuente") or meta.get("ruta") or meta.get("documento") or ""
        posicion = meta.get("posicion") or meta.get("fragmento") or 0
        num_tokens = meta.get("num_tokens") or meta.get("tokens") or len(doc.page_content.split())
        fenomeno = meta.get("fenomeno")
        chunk_id = meta.get("chunk_id") or f"{doc_id}-chunk-{posicion:03d}"

        item = {
            "faiss_id": faiss_id,
            "docstore_id": str(docstore_id),
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "fuente": fuente,
            "formato": formato,
            "fenomeno": fenomeno,
            "posicion": posicion,
            "num_tokens": num_tokens,
            "texto": doc.page_content,
            "metadata": meta,
        }
        fragmentos.append(item)

    # Ordenar por documento y posicion/faiss_id
    fragmentos.sort(key=lambda x: (x["doc_id"], x["posicion"], x["faiss_id"]))
    return fragmentos


def generar_resumen_estadistico(fragmentos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Genera estadísticas detalladas sobre el corpus indexado en FAISS."""
    total_chunks = len(fragmentos)
    if total_chunks == 0:
        return {"total_chunks": 0}

    documentos = set()
    formatos: Dict[str, int] = {}
    fenomenos: Dict[Any, int] = {}
    tokens_totales = 0
    tokens_por_doc: Dict[str, int] = {}
    chunks_por_doc: Dict[str, int] = {}

    for frag in fragmentos:
        doc = frag["doc_id"]
        fmt = frag["formato"]
        fen = frag["fenomeno"] if frag["fenomeno"] is not None else "Sin asignar"
        tokens = frag["num_tokens"]

        documentos.add(doc)
        formatos[fmt] = formatos.get(fmt, 0) + 1
        fenomenos[fen] = fenomenos.get(fen, 0) + 1
        tokens_totales += tokens

        tokens_por_doc[doc] = tokens_por_doc.get(doc, 0) + tokens
        chunks_por_doc[doc] = chunks_por_doc.get(doc, 0) + 1

    promedio_tokens = round(tokens_totales / total_chunks, 2) if total_chunks > 0 else 0

    resumen = {
        "total_chunks": total_chunks,
        "total_documentos": len(documentos),
        "tokens_totales": tokens_totales,
        "promedio_tokens_por_chunk": promedio_tokens,
        "desglose_formatos": formatos,
        "desglose_fenomenos": fenomenos,
        "chunks_por_documento": chunks_por_doc,
        "tokens_por_documento": tokens_por_doc,
    }
    return resumen


def exportar_json(fragmentos: List[Dict[str, Any]], ruta_salida: Path) -> None:
    """Exporta los datos a un archivo JSON con formato legible."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(fragmentos, f, ensure_ascii=False, indent=2)
    print(f"✔ Exportado JSON ({len(fragmentos)} registros) en: {ruta_salida}")


def exportar_jsonl(fragmentos: List[Dict[str, Any]], ruta_salida: Path) -> None:
    """Exporta los datos a formato JSON Lines (1 objeto JSON por línea)."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for frag in fragmentos:
            f.write(json.dumps(frag, ensure_ascii=False) + "\n")
    print(f"✔ Exportado JSONL ({len(fragmentos)} registros) en: {ruta_salida}")


def exportar_csv(fragmentos: List[Dict[str, Any]], ruta_salida: Path) -> None:
    """Exporta los datos a CSV tabular."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    if not fragmentos:
        return

    # Extraer todas las columnas posibles
    columnas_base = ["faiss_id", "doc_id", "chunk_id", "formato", "fenomeno", "posicion", "num_tokens", "fuente", "texto"]
    
    with open(ruta_salida, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas_base, extrasaction="ignore")
        writer.writeheader()
        for frag in fragmentos:
            row = dict(frag)
            # Acortar texto si es para vista previa en CSV si fuera necesario, o dejar texto completo
            writer.writerow(row)

    print(f"✔ Exportado CSV ({len(fragmentos)} registros) en: {ruta_salida}")


def exportar_markdown(fragmentos: List[Dict[str, Any]], resumen: Dict[str, Any], ruta_salida: Path) -> None:
    """Genera un reporte completo en Markdown con estadísticas y muestra de chunks."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Reporte y Resumen de Datos de Índice FAISS")
    lines.append(f"\n> **Índice analizado:** `faiss_index/`  ")
    lines.append(f"> **Total de Chunks:** {resumen['total_chunks']}  ")
    lines.append(f"> **Total de Documentos:** {resumen['total_documentos']}  ")
    lines.append(f"> **Tokens Totales:** {resumen['tokens_totales']:,} (Promedio: {resumen['promedio_tokens_por_chunk']} t/chunk)\n")

    lines.append("---")
    lines.append("## 1. Distribución por Formatos\n")
    lines.append("| Formato | Chunks | Porcentaje |")
    lines.append("| ------- | ------ | ---------- |")
    for fmt, count in sorted(resumen["desglose_formatos"].items(), key=lambda x: x[1], reverse=True):
        pct = round((count / resumen['total_chunks']) * 100, 1)
        lines.append(f"| `{fmt}` | {count} | {pct}% |")

    lines.append("\n## 2. Distribución por Fenómenos Temáticos\n")
    lines.append("| Fenómeno | Chunks | Porcentaje |")
    lines.append("| -------- | ------ | ---------- |")
    for fen, count in sorted(resumen["desglose_fenomenos"].items(), key=lambda x: str(x[0])):
        pct = round((count / resumen['total_chunks']) * 100, 1)
        lines.append(f"| {fen} | {count} | {pct}% |")

    lines.append("\n## 3. Resumen por Documento\n")
    lines.append("| Documento | Chunks | Tokens Totales | Promedio Tokens |")
    lines.append("| --------- | ------ | -------------- | --------------- |")
    for doc, count in sorted(resumen["chunks_por_documento"].items(), key=lambda x: x[1], reverse=True):
        t_total = resumen["tokens_por_documento"][doc]
        t_avg = round(t_total / count, 1)
        lines.append(f"| `{doc}` | {count} | {t_total:,} | {t_avg} |")

    lines.append("\n## 4. Muestra de Fragmentos (Primeros 15)\n")
    for frag in fragmentos[:15]:
        lines.append(f"### Chunk `{frag['chunk_id']}` (FAISS ID: {frag['faiss_id']})")
        lines.append(f"- **Documento:** `{frag['doc_id']}` | **Formato:** `{frag['formato']}` | **Tokens:** {frag['num_tokens']}")
        lines.append(f"- **Fuente:** `{frag['fuente']}`")
        text_preview = frag['texto'][:300].replace("\n", " ") + ("..." if len(frag['texto']) > 300 else "")
        lines.append(f"```text\n{text_preview}\n```\n")

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✔ Exportado Reporte Markdown en: {ruta_salida}")


def ejecutar_busqueda_cli(faiss_dir: Path, query: str, top_k: int = 5) -> None:
    """Ejecuta una búsqueda vectorial de prueba directamente desde la línea de comandos."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        print(f"\n🔍 Buscando en FAISS: '{query}' (top_k={top_k})...\n")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        for rank, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            print(f"--- [Rank {rank}] Score / Distancia: {score:.4f} ---")
            print(f"Doc ID  : {meta.get('doc_id') or meta.get('documento')}")
            print(f"Chunk ID: {meta.get('chunk_id') or meta.get('fragmento')}")
            print(f"Formato : {meta.get('formato') or meta.get('tipo')}")
            print(f"Texto   : {doc.page_content[:250]}...\n")
    except Exception as e:
        print(f"❌ Error al ejecutar búsqueda vectorial: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae, analiza y exporta datos almacenados en faiss_index")
    parser.add_argument("--faiss-dir", type=str, default=str(FAISS_DIR_DEFAULT), help="Ruta al directorio faiss_index")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR_DEFAULT), help="Ruta del directorio de salida")
    parser.add_argument("--format", type=str, choices=["all", "json", "jsonl", "csv", "md"], default="all", help="Formato de exportación")
    parser.add_argument("--stats", action="store_true", help="Mostrar resumen estadístico en terminal")
    parser.add_argument("--search", type=str, help="Ejecutar una consulta semántica de prueba en FAISS")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados para la búsqueda")

    args = parser.parse_args()

    faiss_dir = Path(args.faiss_dir)
    out_dir = Path(args.out_dir)

    print(f"📦 Cargando datos desde: {faiss_dir}")
    fragmentos = cargar_datos_faiss(faiss_dir)
    resumen = generar_resumen_estadistico(fragmentos)

    print(f"\n📊 Se extrajeron {resumen['total_chunks']} chunks de {resumen['total_documentos']} documentos.")

    if args.stats or args.format == "all":
        print("\n--- Resumen Estadístico ---")
        print(f"Chunks Totales      : {resumen['total_chunks']}")
        print(f"Documentos Únicos  : {resumen['total_documentos']}")
        print(f"Tokens Totales      : {resumen['tokens_totales']:,}")
        print(f"Promedio Token/Chunk: {resumen['promedio_tokens_por_chunk']}")
        print(f"Formatos            : {resumen['desglose_formatos']}")
        print(f"Fenómenos           : {resumen['desglose_fenomenos']}")

    fmt = args.format
    if fmt in ["all", "json"]:
        exportar_json(fragmentos, out_dir / "faiss_chunks.json")
    if fmt in ["all", "jsonl"]:
        exportar_jsonl(fragmentos, out_dir / "faiss_chunks.jsonl")
    if fmt in ["all", "csv"]:
        exportar_csv(fragmentos, out_dir / "faiss_chunks.csv")
    if fmt in ["all", "md"]:
        exportar_markdown(fragmentos, resumen, out_dir / "faiss_resumen.md")

    if args.search:
        ejecutar_busqueda_cli(faiss_dir, args.search, top_k=args.top_k)


if __name__ == "__main__":
    main()
