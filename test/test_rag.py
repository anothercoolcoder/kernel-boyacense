import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = "./chroma_db"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-large-instruct"

def probar_busqueda(pregunta: str, k: int = 3):
    if not os.path.exists(DB_DIR):
        print(f"La base de datos '{DB_DIR}' no existe.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    instruccion = "Given a web search query, retrieve relevant passages that answer the query"
    query_formateada = f"Instruct: {instruccion}\nQuery: {pregunta}"

    resultados = vector_store.similarity_search_with_score(query_formateada, k=k)

    print("="*80)
    print(f"PREGUNTA: '{pregunta}'")
    print("="*80)

    for i, (doc, score) in enumerate(resultados, 1):
        print(f"\n--- Resultado #{i} (Score: {score:.4f}) ---")
        print(f"Archivo: {doc.metadata.get('fuente', 'Desconocido')}")
        
        headers = [f"{k}: {v}" for k, v in doc.metadata.items() if k.startswith("Header")]
        if headers:
            print(f"Jerarquía: {' -> '.join(headers)}")
            
        print(f"\nCONTENIDO:\n{doc.page_content.strip()}")
        print("-" * 80)

if __name__ == "__main__":
    pregunta_test = "Escribe aquí la consulta de prueba" 
    probar_busqueda(pregunta_test, k=3)