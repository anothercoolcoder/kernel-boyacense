import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Asegurar que el directorio raíz del proyecto esté en sys.path
RAIZ = Path(__file__).parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from recuperar.recuperar import BuscadorHibrido

# ================== CONFIGURACIÓN ==================
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
    buscador = BuscadorHibrido()
    resultados = {
        "fecha": datetime.now().isoformat(),
        "estrategia": "BuscadorHibrido.buscar",
        "top_k_docs": 3,
        "k": k,
        "total_preguntas": len(preguntas),
        "preguntas": []
    }

    print(f"Procesando {len(preguntas)} preguntas...\n")

    for idx_preg, pregunta in enumerate(preguntas, 1):
        print(f"[{idx_preg}/{len(preguntas)}] {pregunta[:80]}...")

        resultado = buscador.buscar(
            pregunta,
            top_k_docs=3,
            top_k_chunks=k,
        )

        resultados["preguntas"].append({
            "id": idx_preg,
            "pregunta": pregunta,
            "documents": resultado["documents"],
            "fragments": resultado["fragments"],
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
