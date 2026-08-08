import sys
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en sys.path
RAIZ = Path(__file__).parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from recuperar.recuperar import BuscadorHibrido


def probar_busqueda(pregunta: str):
    buscador = BuscadorHibrido()
    resultado = buscador.buscar(pregunta, top_k_docs=3, top_k_chunks=5)

    print("=" * 80)
    print(f"PREGUNTA: '{pregunta}'")
    print("=" * 80)

    print("\n--- TOP DOCUMENTOS (F1@3) ---")
    for doc in resultado["documents"]:
        print(f"Rank {doc['rank']}: {doc['doc_id']}")

    print("\n--- TOP FRAGMENTOS HÍBRIDOS (NDCG@10 / RRF) ---")
    for frag in resultado["fragments"]:
        print(f"\n--- Resultado #{frag['rank']} (Score RRF: {frag['score_rrf']:.5f}) ---")
        print(f"Archivo: {frag.get('fuente', 'Desconocido')}")
        print(f"Doc ID : {frag.get('doc_id', 'Desconocido')}")
        print(f"Chunk  : {frag.get('chunk_id', 'Desconocido')}")
        print(f"Tokens : {frag.get('num_tokens', 0)} | Palabras: {len(frag.get('text', '').split())}")
        print(f"\nCONTENIDO:\n{frag.get('text', '').strip()}")
        print("-" * 80)


if __name__ == "__main__":
    question = input("Ingrese pregunta para hacerle al sistema: ")
    probar_busqueda(question)