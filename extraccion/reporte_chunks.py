"""Genera un Markdown legible con el resultado del pipeline, fragmento a fragmento.

Uso: python reporte_chunks.py [ruta_pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from extraccion import extraer_documento
from fragmentacion import Fragmento, fragmentar_registros

PDF_POR_DEFECTO = Path(__file__).parent / "tallerProbarBestStudent.pdf"


def generar_reporte(ruta_pdf: Path, salida: Path, fragmentos: list[Fragmento] | None = None) -> None:
    """Escribe el Markdown. Si no se pasan ``fragmentos``, corre el pipeline completo."""
    if fragmentos is None:
        fragmentos = fragmentar_registros(extraer_documento(ruta_pdf))

    paginas = len({f["pagina"] for f in fragmentos})
    lineas = [f"# Reporte de chunks — {ruta_pdf.name}\n"]
    lineas.append(f"Páginas: {paginas} · Fragmentos: {len(fragmentos)}\n")

    for f in fragmentos:
        lineas.append(f"## Página {f['pagina']} · Fragmento {f['fragmento']}")
        lineas.append(
            f"*{f['metadata']['tokens']} tokens · origen: {f['metadata'].get('origen_texto', '?')}*\n"
        )
        lineas.append("```")
        lineas.append(f["texto"])
        lineas.append("```\n")

    salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Reporte escrito en: {salida}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_POR_DEFECTO
    generar_reporte(ruta, Path(__file__).parent / f"reporte_{ruta.stem}.md")
