"""Módulo de Construcción del Grafo de Conocimiento a partir de metadatos de fragmentos.

Cumple estrictamente con las reglas de negocio de CODEFEST AD ASTRA 2026:
- Co-ocurrencia de entidades dentro del mismo fragmento (chunk_id, doc_id).
- Guarda obligatoriamente doc_id y chunk_id en las propiedades de cada arista.
- Exporta en formato GraphML (`grafo.graphml`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from tqdm import tqdm

from grafo.extractor import ExtractorEntidades, normalizar_entidad

logger = logging.getLogger(__name__)



class ConstructorGrafo:
    """Construye un grafo de conocimiento NetworkX a partir de `metadata.jsonl`."""

    def __init__(self, extractor: ExtractorEntidades | None = None) -> None:
        self.extractor = extractor or ExtractorEntidades()
        self.graph = nx.Graph()

    def construir_desde_metadata(
        self,
        metadata_path: Path | str,
        limit_chunks: int | None = None
    ) -> nx.Graph:
        """Procesa `metadata.jsonl`, extrae entidades y construye el grafo de co-ocurrencia."""
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de metadatos en: {path}")

        logger.info(f"Procesando metadatos para grafo desde {path}...")
        self.graph = nx.Graph()

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if limit_chunks:
            lines = lines[:limit_chunks]

        for line in tqdm(lines, desc="Extrayendo entidades para Grafo"):
            if not line.strip():
                continue
            chunk_data = json.loads(line.strip())

            doc_id = str(chunk_data.get("doc_id", ""))
            chunk_id = str(chunk_data.get("chunk_id", ""))
            texto = chunk_data.get("texto", "")

            if not doc_id or not chunk_id or not texto:
                continue

            entidades_extraidas = self.extractor.extraer(texto)
            if not entidades_extraidas:
                continue

            # Agrupar entidades únicas en este fragmento
            entidades_chunk: Dict[str, str] = {}
            for ent in entidades_extraidas:
                nombre_norm = normalizar_entidad(ent["text"])
                if len(nombre_norm) >= 2:
                    entidades_chunk[nombre_norm] = ent.get("label", "entidad")

            # 1. Agregar / actualizar nodos
            for nombre, label in entidades_chunk.items():
                if self.graph.has_node(nombre):
                    self.graph.nodes[nombre]["frecuencia"] += 1
                else:
                    self.graph.add_node(
                        nombre,
                        label=label,
                        frecuencia=1
                    )

            # 2. Agregar / actualizar aristas por co-ocurrencia dentro del fragmento
            nodos = list(entidades_chunk.keys())
            for i in range(len(nodos)):
                for j in range(i + 1, len(nodos)):
                    u, v = nodos[i], nodos[j]

                    if self.graph.has_edge(u, v):
                        edge_data = self.graph[u][v]
                        edge_data["weight"] += 1
                        
                        # Actualizar lista de evidencias sin duplicados
                        evidencias = json.loads(edge_data.get("evidencias_json", "[]"))
                        evidencia_nueva = {"doc_id": doc_id, "chunk_id": chunk_id}
                        if evidencia_nueva not in evidencias:
                            evidencias.append(evidencia_nueva)
                        
                        edge_data["evidencias_json"] = json.dumps(evidencias)
                        # Mantener doc_id y chunk_id más recientes/representativos
                        doc_ids = set(edge_data["doc_id"].split(","))
                        doc_ids.add(doc_id)
                        edge_data["doc_id"] = ",".join(doc_ids)

                        chunk_refs = set(edge_data["chunk_id"].split(","))
                        chunk_refs.add(f"{doc_id}:{chunk_id}")
                        edge_data["chunk_id"] = ",".join(chunk_refs)
                    else:
                        evidencias = [{"doc_id": doc_id, "chunk_id": chunk_id}]
                        self.graph.add_edge(
                            u,
                            v,
                            weight=1,
                            relacion="co_ocurre_con",
                            doc_id=doc_id,
                            chunk_id=f"{doc_id}:{chunk_id}",
                            evidencias_json=json.dumps(evidencias)
                        )

            count += 1

        logger.info(f"Grafo construido con {self.graph.number_of_nodes()} nodos y {self.graph.number_of_edges()} aristas.")
        return self.graph

    def exportar_graphml(self, output_path: Path | str) -> Path:
        """Exporta el grafo al formato `grafo.graphml` exigido por las especificaciones."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, str(out))
        logger.info(f"Grafo exportado exitosamente a {out}")
        return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Construir Grafo de Conocimiento")
    parser.add_argument(
        "--metadata",
        type=str,
        default="base_vectorial/encoder_multilingual-e5-large-instruct-prueba/metadata.jsonl",
        help="Ruta al archivo metadata.jsonl"
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="entrega/grafo/grafo.graphml",
        help="Ruta de destino del archivo GraphML"
    )
    args = parser.parse_args()

    builder = ConstructorGrafo()
    builder.construir_desde_metadata(args.metadata)
    builder.exportar_graphml(args.salida)
