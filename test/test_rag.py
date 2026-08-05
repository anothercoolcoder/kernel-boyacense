import os
import json
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_DIR = "./base_vectorial/encoder_multilingual-e5-large-instruct"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-large-instruct"

def probar_busqueda(pregunta: str, k: int = 3):
    index_path = os.path.join(FAISS_DIR, "index.faiss")
    metadata_path = os.path.join(FAISS_DIR, "metadata.jsonl")

    if not os.path.exists(index_path):
        print(f"El archivo '{index_path}' no existe. Ejecuta primero el pipeline (python main.py)")
        return

    # Cargar índice FAISS nativo
    index = faiss.read_index(index_path)

    # Cargar metadata
    metadatos = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadatos.append(json.loads(line))

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    instruccion = "Given a web search query, retrieve relevant passages that answer the query"
    query_formateada = f"Instruct: {instruccion}\nQuery: {pregunta}"

    query_vector = np.array([embeddings.embed_query(query_formateada)], dtype=np.float32)
    scores, indices = index.search(query_vector, k)

    print("="*80)
    print(f"PREGUNTA: '{pregunta}'")
    print("="*80)

    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx < 0 or idx >= len(metadatos):
            continue
        meta = metadatos[idx]
        print(f"\n--- Resultado #{i} (Score Similitud IP: {score:.4f}) ---")
        print(f"Archivo: {meta.get('fuente', 'Desconocido')}")
        print(f"Doc ID : {meta.get('doc_id', 'Desconocido')}")
        print(f"Chunk  : {meta.get('chunk_id', 'Desconocido')}")
        print(f"Formato: {meta.get('formato', 'Desconocido')}")
        print(f"\nCONTENIDO:\n{meta.get('texto', '').strip()}")
        print("-" * 80)

if __name__ == "__main__":
    question = input("Ingrese pregunta para hacerle al sistema: ")
    probar_busqueda(question, k=5)