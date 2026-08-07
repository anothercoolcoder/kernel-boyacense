import os
import json
import numpy as np
import faiss
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

# ================== CONFIGURACIÓN ==================
FAISS_DIR = "./base_vectorial/encoder_multilingual-e5-large-instruct"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-large-instruct"
ARCHIVO_PREGUNTAS = "preguntas.txt"          # o "preguntas.md"
ARCHIVO_SALIDA = "resultados_rag.json"       # salida estructurada para análisis
K = 5                                        # número de chunks a recuperar

def cargar_preguntas(ruta: str) -> list[str]:
    """Lee preguntas, una por línea. Ignora líneas vacías y comentarios (#)."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo de preguntas: {ruta}")
    
    preguntas = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            preguntas.append(linea)
    return preguntas

def probar_busqueda_batch(preguntas: list[str], k: int = 5):
    index_path = os.path.join(FAISS_DIR, "index.faiss")
    metadata_path = os.path.join(FAISS_DIR, "metadata.jsonl")

    if not os.path.exists(index_path):
        print(f"El archivo '{index_path}' no existe. Ejecuta primero el pipeline.")
        return

    # Cargar índice y metadata
    index = faiss.read_index(index_path)
    metadatos = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadatos.append(json.loads(line))

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    instruccion = "Given a web search query, retrieve relevant passages that answer the query"
    resultados = {
        "fecha": datetime.now().isoformat(),
        "modelo": MODELO_EMBEDDINGS,
        "k": k,
        "total_preguntas": len(preguntas),
        "preguntas": []
    }

    print(f"Procesando {len(preguntas)} preguntas...\n")

    for idx_preg, pregunta in enumerate(preguntas, 1):
        print(f"[{idx_preg}/{len(preguntas)}] {pregunta[:80]}...")

        query_formateada = f"Instruct: {instruccion}\nQuery: {pregunta}"
        query_vector = np.array([embeddings.embed_query(query_formateada)], dtype=np.float32)
        scores, indices = index.search(query_vector, k)

        chunks_recuperados = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(metadatos):
                continue
            meta = metadatos[idx]
            chunks_recuperados.append({
                "rank": rank,
                "score": float(score),
                "fuente": meta.get("fuente", "Desconocido"),
                "doc_id": meta.get("doc_id", "Desconocido"),
                "chunk_id": meta.get("chunk_id", "Desconocido"),
                "formato": meta.get("formato", "Desconocido"),
                "texto": meta.get("texto", "").strip()
            })

        resultados["preguntas"].append({
            "id": idx_preg,
            "pregunta": pregunta,
            "chunks": chunks_recuperados
        })

    # Guardar JSON completo
    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Resultados guardados en: {ARCHIVO_SALIDA}")
    print(f"Total de preguntas procesadas: {len(preguntas)}")
    return resultados

if __name__ == "__main__":
    try:
        lista_preguntas = cargar_preguntas(ARCHIVO_PREGUNTAS)
        if not lista_preguntas:
            print("No se encontraron preguntas válidas en el archivo.")
        else:
            probar_busqueda_batch(lista_preguntas, k=K)
    except Exception as e:
        print(f"Error: {e}")