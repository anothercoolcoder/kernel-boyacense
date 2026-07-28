"""
Normalización de texto — limpieza y estandarización a UTF-8/NFC.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normaliza texto crudo para procesamiento downstream.

    Operaciones (en orden):
    1. Strip de BOM (Byte Order Mark)
    2. Decodificación/recodificación explícita a UTF-8
    3. Normalización Unicode NFC (forma compuesta canónica)
    4. Reemplazo de comillas tipográficas y guiones especiales
    5. Eliminación de caracteres de control (excepto \\n, \\t)
    6. Colapso de whitespace múltiple
    7. Strip final
    """

    # Mapeo de caracteres tipográficos a ASCII
    _TYPOGRAPHIC_MAP: dict[str, str] = {
        "\u2018": "'",   # comilla simple izquierda
        "\u2019": "'",   # comilla simple derecha
        "\u201C": '"',   # comilla doble izquierda
        "\u201D": '"',   # comilla doble derecha
        "\u2013": "-",   # guión medio (en dash)
        "\u2014": "-",   # guión largo (em dash)
        "\u2026": "...", # puntos suspensivos
        "\u00A0": " ",   # espacio de no separación
    }

    # Regex para caracteres de control (excepto \n y \t)
    _CONTROL_CHARS_RE = re.compile(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"
    )

    # Regex para whitespace múltiple (dentro de una línea)
    _MULTI_SPACE_RE = re.compile(r"[^\S\n]+")

    # Regex para líneas vacías múltiples
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

    def normalize(self, text: str) -> str:
        """
        Aplica todas las transformaciones de normalización al texto.

        Parameters
        ----------
        text : str
            Texto crudo a normalizar.

        Returns
        -------
        str
            Texto normalizado y limpio.
        """
        original_len = len(text)

        # 1. Strip BOM
        if text.startswith("\ufeff"):
            text = text[1:]

        # 2. Forzar UTF-8 — eliminar caracteres que no se pueden codificar
        text = text.encode("utf-8", errors="replace").decode("utf-8")

        # 3. Normalización Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # 4. Reemplazo de caracteres tipográficos
        for original, replacement in self._TYPOGRAPHIC_MAP.items():
            text = text.replace(original, replacement)

        # 5. Eliminar caracteres de control
        text = self._CONTROL_CHARS_RE.sub("", text)

        # 6. Colapso de whitespace
        text = self._MULTI_SPACE_RE.sub(" ", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)

        # 7. Strip final
        text = text.strip()

        final_len = len(text)
        if original_len != final_len:
            logger.debug(
                "  Normalización: %d → %d caracteres (Δ%d)",
                original_len,
                final_len,
                final_len - original_len,
            )

        return text
