"""Script principal para construir y exportar el Grafo de Conocimiento entregable.

Lee los metadatos de los fragmentos, ejecuta el NER encoder (GLiNER / fallback),
construye las aristas de co-ocurrencia con metadatos doc_id y chunk_id,
y exporta el resultado a `entrega/grafo/grafo.graphml`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from grafo.builder import ConstructorGrafo


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir Grafo de Conocimiento (CODEFEST AD ASTRA 2026)")
    parser.add_argument(
        "--metadata",
        type=str,
        default="base_vectorial/encoder_multilingual-e5-large-instruct-prueba/metadata.jsonl",
        help="Ruta al archivo metadata.jsonl con los fragmentos."
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="entrega/grafo/grafo.graphml",
        help="Ruta donde se guardará el grafo en formato GraphML."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite opcional de fragmentos a procesar (para pruebas rápidas)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de lote para extracción (aumentar en GPUs con mayor VRAM)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="urchade/gliner_small-v2.1",
        help="Nombre del modelo GLiNER a utilizar."
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    salida_path = Path(args.salida).resolve()

    if not metadata_path.exists():
        print(f"[!] Error: No se encontró el archivo de metadatos en {metadata_path}")
        print("    Asegúrate de que la indexación se haya ejecutado primero o especifica --metadata con la ruta correcta.")
        sys.exit(1)

    print(f"[*] Iniciando construcción del grafo desde: {metadata_path}")
    builder = ConstructorGrafo(model_name=args.model_name)
    g = builder.construir_desde_metadata(metadata_path, limit_chunks=args.limit, batch_size=args.batch_size)

    print(f"[*] Nodos extraídos: {g.number_of_nodes()} | Aristas: {g.number_of_edges()}")

    salida_path.parent.mkdir(parents=True, exist_ok=True)
    builder.exportar_graphml(salida_path)

    print(f"[✓] Grafo exportado exitosamente en: {salida_path} ({salida_path.stat().st_size} bytes)")



if __name__ == "__main__":
    main()
