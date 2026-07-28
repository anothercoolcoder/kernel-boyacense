"""
Chunking con respeto a límites de oración.

Garantiza que ninguna oración se divida entre chunks diferentes.
Utiliza pysbd para la detección de oraciones y el tokenizer del
modelo E5 para contar tokens reales.
"""

import logging

import pysbd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Modelo cuyo tokenizer se usa para contar tokens
_DEFAULT_MODEL = "intfloat/multilingual-e5-base"


class SentenceChunker:
    """
    Divide texto en chunks respetando límites de oración.

    Cada chunk contiene oraciones completas y no excede
    ``max_tokens`` tokens del tokenizer del modelo de embeddings.

    Parameters
    ----------
    max_tokens : int
        Máximo de tokens por chunk (sin contar [CLS]/[SEP]).
        Default: 510 (512 - 2 tokens especiales).
    overlap_sentences : int
        Cantidad de oraciones del final del chunk anterior que se
        repiten al inicio del siguiente (solapamiento).
        Default: 1.
    language : str
        Idioma para la detección de oraciones.
        Default: "es" (español).
    model_name : str
        Nombre del modelo cuyo tokenizer se usa para contar tokens.
    """

    def __init__(
        self,
        max_tokens: int = 510,
        overlap_sentences: int = 1,
        language: str = "es",
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.language = language

        logger.info("⚙️  Cargando tokenizer de '%s'…", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("  └─ Tokenizer cargado correctamente.")

        self._segmenter = pysbd.Segmenter(
            language=language,
            clean=False,  # no modificar el texto, solo segmentar
        )

    def _count_tokens(self, text: str) -> int:
        """Cuenta tokens usando el tokenizer del modelo."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _segment_sentences(self, text: str) -> list[str]:
        """Segmenta texto en oraciones usando pysbd."""
        sentences = self._segmenter.segment(text)
        # Limpiar oraciones vacías
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> list[dict]:
        """
        Divide el texto en chunks de oraciones completas.

        Parameters
        ----------
        text : str
            Texto normalizado a dividir.

        Returns
        -------
        list[dict]
            Lista de diccionarios con claves:
            - "text": texto del chunk
            - "num_sentences": cantidad de oraciones
            - "num_tokens": cantidad de tokens
        """
        sentences = self._segment_sentences(text)
        if not sentences:
            logger.warning("  ⚠️  Texto vacío o sin oraciones detectables.")
            return []

        logger.info(
            "  ├─ Oraciones detectadas: %d (idioma: %s)",
            len(sentences),
            self.language,
        )

        chunks: list[dict] = []
        current_sentences: list[str] = []
        current_tokens: int = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Caso especial: una sola oración excede max_tokens
            if sentence_tokens > self.max_tokens:
                # Cerrar chunk actual si tiene contenido
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append({
                        "text": chunk_text,
                        "num_sentences": len(current_sentences),
                        "num_tokens": current_tokens,
                    })
                    current_sentences = []
                    current_tokens = 0

                # La oración larga va sola (no la dividimos, es el requisito)
                logger.warning(
                    "  ├─ ⚠️  Oración con %d tokens excede límite de %d. "
                    "Se incluye completa en un chunk individual.",
                    sentence_tokens,
                    self.max_tokens,
                )
                chunks.append({
                    "text": sentence,
                    "num_sentences": 1,
                    "num_tokens": sentence_tokens,
                })
                continue

            # ¿Agregar esta oración excede el límite?
            projected_tokens = self._count_tokens(
                " ".join(current_sentences + [sentence])
            )

            if projected_tokens > self.max_tokens and current_sentences:
                # Cerrar el chunk actual
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "text": chunk_text,
                    "num_sentences": len(current_sentences),
                    "num_tokens": current_tokens,
                })

                # Overlap: tomar las últimas N oraciones
                if self.overlap_sentences > 0:
                    overlap = current_sentences[-self.overlap_sentences:]
                    current_sentences = overlap
                    current_tokens = self._count_tokens(" ".join(current_sentences))
                else:
                    current_sentences = []
                    current_tokens = 0

            # Agregar la oración al chunk actual
            current_sentences.append(sentence)
            current_tokens = self._count_tokens(" ".join(current_sentences))

        # Flush del último chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text,
                "num_sentences": len(current_sentences),
                "num_tokens": current_tokens,
            })

        logger.info(
            "  └─ Chunks generados: %d (tokens promedio: %.0f)",
            len(chunks),
            sum(c["num_tokens"] for c in chunks) / max(len(chunks), 1),
        )

        return chunks
