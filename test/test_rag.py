import os
import json
import argparse
from datetime import datetime, timezone

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_DIR = "./faiss_index"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-large-instruct"
SALIDA_JSONL = "./resultados.jsonl"


def cargar_vector_store(faiss_dir: str = FAISS_INDEX_DIR) -> FAISS:
    """Carga el índice FAISS persistido junto con el modelo de embeddings."""
    if not os.path.exists(faiss_dir):
        raise FileNotFoundError(
            f"El directorio '{faiss_dir}' no existe. Ejecuta primero indexar.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return FAISS.load_local(
        faiss_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )


def buscar(vector_store: FAISS, pregunta: str, k: int = 3):
    """Ejecuta la búsqueda FAISS (similaridad + score) para una pregunta dada."""
    instruccion = "Given a web search query, retrieve relevant passages that answer the query"
    query_formateada = f"Instruct: {instruccion}\nQuery: {pregunta}"
    return vector_store.similarity_search_with_score(query_formateada, k=k)


def resultados_a_registros(pregunta: str, resultados) -> list[dict]:
    """Convierte los resultados FAISS al esquema de salida resultados.jsonl."""
    registros = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for i, (doc, score) in enumerate(resultados, 1):
        headers = {k: v for k, v in doc.metadata.items() if k.startswith("Header")}
        registros.append({
            "pregunta": pregunta,
            "rank": i,
            "score_l2": float(score),
            "doc_id": doc.metadata.get("doc_id", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "fuente": doc.metadata.get("fuente", "Desconocido"),
            "jerarquia": headers,
            "contenido": doc.page_content.strip(),
            "timestamp": timestamp,
        })
    return registros


def guardar_jsonl(registros: list[dict], salida: str = SALIDA_JSONL, modo: str = "a"):
    """Agrega (o sobreescribe) los registros en resultados.jsonl."""
    with open(salida, modo, encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def imprimir_resultados(pregunta: str, resultados):
    print("=" * 80)
    print(f"PREGUNTA: '{pregunta}'")
    print("=" * 80)

    for i, (doc, score) in enumerate(resultados, 1):
        print(f"\n--- Resultado #{i} (Score Distancia L2: {score:.4f}) ---")
        print(f"Archivo: {doc.metadata.get('fuente', 'Desconocido')}")

        headers = [f"{k}: {v}" for k, v in doc.metadata.items() if k.startswith("Header")]
        if headers:
            print(f"Jerarquía: {' -> '.join(headers)}")

        print(f"\nCONTENIDO:\n{doc.page_content.strip()}")
        print("-" * 80)


def probar_busqueda(pregunta: str, k: int = 3, guardar: bool = True, salida: str = SALIDA_JSONL, sobrescribir: bool = False):
    vector_store = cargar_vector_store()
    resultados = buscar(vector_store, pregunta, k=k)

    imprimir_resultados(pregunta, resultados)

    if guardar:
        registros = resultados_a_registros(pregunta, resultados)
        modo = "w" if sobrescribir else "a"
        guardar_jsonl(registros, salida=salida, modo=modo)
        print(f"\n✔ {len(registros)} resultado(s) guardado(s) en: '{salida}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda semántica sobre el índice FAISS (etapa de recuperación del pipeline RAG).")
    parser.add_argument(
        "pregunta",
        nargs="?",
        default="¿Qué clínicas y entidades bancarias se encuentran indexadas en la zona?",
        help="Pregunta en texto natural a consultar."
    )
    parser.add_argument("-k", type=int, default=3, help="Número de resultados a recuperar (default: 3).")
    parser.add_argument("--salida", type=str, default=SALIDA_JSONL, help="Ruta del archivo resultados.jsonl.")
    parser.add_argument("--no-guardar", action="store_true", help="No escribir resultados.jsonl, solo imprimir en consola.")
    parser.add_argument("--sobrescribir", action="store_true", help="Sobrescribe resultados.jsonl en lugar de agregar al final.")

    args = parser.parse_args()

    try:
        probar_busqueda(
            args.pregunta,
            k=args.k,
            guardar=not args.no_guardar,
            salida=args.salida,
            sobrescribir=args.sobrescribir,
        )
    except FileNotFoundError as e:
        print(e)