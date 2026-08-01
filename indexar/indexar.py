import os
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
CARPETA_ENTRADA = "./salida"
DB_DIR = "./chroma_db"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-large-instruct"

def cargar_y_procesar_markdowns(directorio_salida):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )

    archivos_md = list(Path(directorio_salida).rglob("*.md"))
    print(f"📂 Archivos .md encontrados en '{directorio_salida}': {len(archivos_md)}")

    documentos_procesados = []

    for ruta_archivo in archivos_md:
        try:
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                contenido = f.read()

            chunks_md = markdown_splitter.split_text(contenido)
            chunks_finales = text_splitter.split_documents(chunks_md)

            for chunk in chunks_finales:
                chunk.metadata["fuente"] = str(ruta_archivo.name)
                chunk.metadata["ruta_completa"] = str(ruta_archivo)

            documentos_procesados.extend(chunks_finales)
            print(f"  ✓ Procesado: {ruta_archivo.name} ({len(chunks_finales)} chunks)")

        except Exception as e:
            print(f"  ❌ Error procesando {ruta_archivo.name}: {e}")

    return documentos_procesados


def main():
    if not os.path.exists(CARPETA_ENTRADA):
        print(f"❌ La carpeta '{CARPETA_ENTRADA}' no existe. Créala y añade tus archivos .md")
        return

    print("\n--- 1. Cargando y dividiendo archivos Markdown ---")
    chunks = cargar_y_procesar_markdowns(CARPETA_ENTRADA)

    if not chunks:
        print("⚠️ No se encontraron chunks para procesar.")
        return

    print(f"\nTotal de chunks listos: {len(chunks)}")

    print("\n--- 2. Cargando Modelo multilingual-e5-large-instruct ---")
    # Configuración específica para el modelo e5-instruct
    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={'device': 'cpu'},  # Cambia a 'cuda' si usas GPU Nvidia
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )

    print("\n--- 3. Generando Embeddings e Indexando en ChromaDB ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"✅ ¡Indexación completada exitosamente! Guardado en: '{DB_DIR}'")

if __name__ == "__main__":
    main()