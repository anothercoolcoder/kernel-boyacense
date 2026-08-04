import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_DIR = "./faiss_index"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-large-instruct"

def probar_busqueda(pregunta: str, k: int = 3):
    if not os.path.exists(FAISS_INDEX_DIR):
        print(f"El directorio '{FAISS_INDEX_DIR}' no existe. Ejecuta primero indexar.py")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Cargar el índice FAISS previamente guardado
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    instruccion = "Given a web search query, retrieve relevant passages that answer the query"
    query_formateada = f"Instruct: {instruccion}\nQuery: {pregunta}"

    resultados = vector_store.similarity_search_with_score(query_formateada, k=k)

    print("="*80)
    print(f"PREGUNTA: '{pregunta}'")
    print("="*80)

    for i, (doc, score) in enumerate(resultados, 1):
        print(f"\n--- Resultado #{i} (Score Distancia L2: {score:.4f}) ---")
        print(f"Archivo: {doc.metadata.get('fuente', 'Desconocido')}")
        
        headers = [f"{k}: {v}" for k, v in doc.metadata.items() if k.startswith("Header")]
        if headers:
            print(f"Jerarquía: {' -> '.join(headers)}")
            
        print(f"\nCONTENIDO:\n{doc.page_content.strip()}")
        print("-" * 80)

if __name__ == "__main__":
    question = input("Ingrese pregunta para hacerle al sistema: ")
    probar_busqueda(question, k=5)