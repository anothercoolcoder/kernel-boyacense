"""Módulo de Recuperación Basada en Grafo de Conocimiento (1-Hop Subgraph Retrieval).

Cumple estrictamente con el Paso 4 de las instrucciones:
1. Detecta entidades en la consulta con el mismo extractor NER.
2. Consulta el grafo: identifica nodos coincidentes y sus vecinos directos de primer orden (1-hop).
3. Recupera los fragmentos (doc_id, chunk_id) asociados a las aristas relevantes.
4. Genera un score por candidato para fusionar en el pool RRF.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from grafo.extractor import ExtractorEntidades, normalizar_entidad

logger = logging.getLogger(__name__)


class GrafoRecuperador:
    """Motor de búsqueda topológica sobre `grafo.graphml`."""

    def __init__(
        self,
        graphml_path: Path | str | None = None,
        extractor: ExtractorEntidades | None = None
    ) -> None:
        self.extractor = extractor or ExtractorEntidades()
        self.graph: nx.Graph | None = None

        if graphml_path and Path(graphml_path).exists():
            self.cargar_grafo(graphml_path)

    def cargar_grafo(self, graphml_path: Path | str) -> None:
        """Carga el grafo exportado en formato GraphML."""
        path = Path(graphml_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo GraphML no encontrado en: {path}")

        logger.info(f"Cargando grafo desde {path}...")
        self.graph = nx.read_graphml(str(path))
        logger.info(f"Grafo cargado exitosamente: {self.graph.number_of_nodes()} nodos, {self.graph.number_of_edges()} aristas.")

    def recuperar_fragmentos(
        self,
        query: str,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """Busca fragmentos relevantes en el grafo basándose en las entidades de la consulta.
        
        Retorna lista de diccionarios:
        [{"doc_id": str, "chunk_id": str, "score": float, "entidades_coincidentes": list[str]}]
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            logger.warning("El grafo está vacío o no ha sido cargado.")
            return []

        # 1. Extraer entidades de la consulta
        entidades_query = self.extractor.extraer(query)
        nombres_query = [normalizar_entidad(e["text"]) for e in entidades_query if len(normalizar_entidad(e["text"])) >= 2]

        if not nombres_query:
            # Fallback simple: buscar coincidencia directa de palabras clave con nombres de nodos
            palabras_query = [normalizar_entidad(p) for p in query.split() if len(p) >= 4]
            nombres_query = [n for n in self.graph.nodes() if any(p in n for p in palabras_query)]

        if not nombres_query:
            return []

        # 2. Identificar nodos coincidentes en el grafo
        nodos_coincidentes: Set[str] = set()
        for nombre in nombres_query:
            for nodo in self.graph.nodes():
                if nombre.lower() in str(nodo).lower() or str(nodo).lower() in nombre.lower():
                    nodos_coincidentes.add(str(nodo))

        if not nodos_coincidentes:
            return []

        # 3. Explorar aristas asociadas a los nodos coincidentes y a sus vecinos (1-hop)
        chunk_scores: Dict[Tuple[str, str], float] = defaultdict(float)
        chunk_entidades: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for nodo in nodos_coincidentes:
            # Aristas directas del nodo
            for u, v, edge_data in self.graph.edges(nodo, data=True):
                weight = float(edge_data.get("weight", 1.0))
                evidencias_raw = edge_data.get("evidencias_json", "[]")
                try:
                    evidencias = json.loads(evidencias_raw)
                except Exception:
                    evidencias = []

                if not evidencias and "doc_id" in edge_data and "chunk_id" in edge_data:
                    doc_id = str(edge_data["doc_id"])
                    chunk_ref = str(edge_data["chunk_id"])
                    chunk_id = chunk_ref.split(":")[-1] if ":" in chunk_ref else chunk_ref
                    evidencias = [{"doc_id": doc_id, "chunk_id": chunk_id}]

                for ev in evidencias:
                    d_id = str(ev.get("doc_id", ""))
                    c_id = str(ev.get("chunk_id", ""))
                    if d_id and c_id:
                        key = (d_id, c_id)
                        chunk_scores[key] += weight * 2.0  # Ponderación mayor por coincidencia directa
                        chunk_entidades[key].add(str(nodo))

            # Vecinos de primer orden (1-hop neighbors)
            for vecino in self.graph.neighbors(nodo):
                for u, v, edge_data in self.graph.edges(vecino, data=True):
                    weight = float(edge_data.get("weight", 1.0))
                    evidencias_raw = edge_data.get("evidencias_json", "[]")
                    try:
                        evidencias = json.loads(evidencias_raw)
                    except Exception:
                        evidencias = []

                    for ev in evidencias:
                        d_id = str(ev.get("doc_id", ""))
                        c_id = str(ev.get("chunk_id", ""))
                        if d_id and c_id:
                            key = (d_id, c_id)
                            chunk_scores[key] += weight * 0.5  # Ponderación menor por vecino 1-hop
                            chunk_entidades[key].add(str(vecino))

        # 4. Construir y ordenar el ranking de fragmentos
        resultados = []
        for (d_id, c_id), score in chunk_scores.items():
            resultados.append({
                "doc_id": d_id,
                "chunk_id": c_id,
                "score": float(score),
                "entidades_coincidentes": list(chunk_entidades[(d_id, c_id)])
            })

        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]
