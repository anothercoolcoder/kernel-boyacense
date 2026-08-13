"""Módulo de Reconocimiento de Entidades Nombradas (NER) basado exclusivamente en modelos Encoders (GLiNER).

Cumple con la restricción estricta del CODEFEST AD ASTRA 2026:
- 100% libre de modelos generativos / decoders (GPT, LLaMA, Gemini, Claude).
- Utiliza GLiNER (encoder tipo BERT/DeBERTa) para NER zero-shot multilingüe.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

# Etiquetas por defecto relevantes para el corpus militar, espacial y tecnológico
DEFAULT_LABELS = [
    "país",
    "organización",
    "tecnología militar",
    "agencia espacial",
    "persona",
    "convenio internacional",
    "sistema autónomo",
    "evento"
]

def normalizar_entidad(texto: str) -> str:
    """Normaliza el texto de una entidad para consolidar nodos equivalentes."""
    clean = texto.strip()
    clean = clean.replace(".", "").replace(",", "")
    return clean.strip()


class ExtractorEntidades:
    """Extractor de entidades utilizando GLiNER (Zero-Shot Encoder NER)."""

    def __init__(
        self,
        model_name: str = "urchade/gliner_small-v2.1",
        labels: List[str] | None = None,
        threshold: float = 0.35,
        use_fallback: bool = True
    ) -> None:
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold
        self.use_fallback = use_fallback
        self.model = None

        try:
            import torch
            from gliner import GLiNER
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Cargando modelo GLiNER: {model_name} en dispositivo {device}")
            
            # gliner_small cabe perfectamente en 4GB incluso en fp32. 
            self.model = GLiNER.from_pretrained(model_name).to(device)
            
            # Aumentar el límite a 512 tokens para evitar truncamientos
            self.model.config.max_len = 512
        except Exception as e:
            logger.warning(f"No se pudo cargar GLiNER ({e}). Usando modo fallback de respaldo.")
            self.model = None

    def extraer(self, texto: str, threshold: float | None = None) -> List[Dict[str, Any]]:
        """Extrae entidades nombradas de un fragmento de texto."""
        if not texto or not texto.strip():
            return []

        th = threshold if threshold is not None else self.threshold

        if self.model is not None:
            try:
                entities = self.model.predict_entities(texto, self.labels, threshold=th)
                res = []
                for ent in entities:
                    res.append({
                        "text": ent.get("text", "").strip(),
                        "label": ent.get("label", "").lower(),
                        "score": float(ent.get("score", 1.0))
                    })
                return res
            except Exception as e:
                logger.error(f"Error al ejecutar GLiNER: {e}")

        if self.use_fallback:
            return self._extraer_fallback(texto)

        return []

    def extraer_batch(self, textos: List[str], threshold: float | None = None) -> List[List[Dict[str, Any]]]:
        """Extrae entidades de una lista de fragmentos (batch)."""
        if not textos:
            return []

        th = threshold if threshold is not None else self.threshold

        if self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    # El límite max_len=512 ya fue configurado en la carga
                    batch_entities = self.model.batch_predict_entities(textos, self.labels, threshold=th)
                
                res_final = []
                for entities in batch_entities:
                    res = []
                    for ent in entities:
                        res.append({
                            "text": ent.get("text", "").strip(),
                            "label": ent.get("label", "").lower(),
                            "score": float(ent.get("score", 1.0))
                        })
                    res_final.append(res)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                return res_final
            except Exception as e:
                logger.error(f"Error al ejecutar GLiNER batch: {e}")

        # Fallback o si hay error, procesar secuencialmente (o con fallback regex)
        return [self.extraer(texto, threshold=th) for texto in textos]

    def _extraer_fallback(self, texto: str) -> List[Dict[str, Any]]:
        """Extractor heurístico de respaldo (basado en expresiones regulares / mayúsculas)."""
        entidades: List[Dict[str, Any]] = []
        vistos: Set[str] = set()

        # Siglas y Nombres Propios en mayúscula (ej: EE.UU., OTAN, NASA, DARPA, Colombia, ONU)
        patron_siglas = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,}(?:\.[A-ZÁÉÍÓÚÑ]{1,2}\.?)?)\b")
        patron_nombres = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\b")

        for m in patron_siglas.finditer(texto):
            val = m.group(1).strip()
            if len(val) >= 2 and val not in vistos:
                vistos.add(val)
                entidades.append({"text": val, "label": "organización/sigla", "score": 0.8})

        for m in patron_nombres.finditer(texto):
            val = m.group(1).strip()
            # Filtrar palabras comunes al inicio de oración
            if val not in vistos and not val.startswith(("El ", "La ", "Los ", "Las ", "Este ", "Esta ")):
                vistos.add(val)
                entidades.append({"text": val, "label": "entidad/nombre_propio", "score": 0.7})

        return entidades
