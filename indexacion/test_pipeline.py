"""Prueba end-to-end del pipeline: extracción -> fragmentación.

Corre las autopruebas de cada etapa y encadena ambas sobre archivos reales,
para detectar problemas de integración que las autopruebas aisladas (con
datos sintéticos) no ven. No asume ningún formato ni archivo concreto.

Uso:
    python test_pipeline.py                  # prueba todo archivo soportado en indexacion/
    python test_pipeline.py a.pdf b.png ...   # prueba solo los archivos dados
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows es cp1252

from extraccion import _autoprueba as autoprueba_extraccion
from extraccion import extraer_documento, formatos_soportados
from fragmentacion import MAX_TOKENS, _autoprueba as autoprueba_fragmentacion
from fragmentacion import fragmentar_registros
from reporte_chunks import generar_reporte

CARPETA = Path(__file__).parent


def _archivos_a_probar() -> list[Path]:
    """Archivos pasados por CLI, o todo archivo con extractor registrado en la carpeta."""
    if len(sys.argv) > 1:
        return [Path(a) for a in sys.argv[1:]]
    soportadas = set(formatos_soportados())
    return sorted(
        p
        for p in CARPETA.iterdir()
        if p.is_file() and p.suffix.lower() in soportadas and not p.name.startswith("reporte_")
    )


def test_etapas_aisladas() -> None:
    print("== Autoprueba extraccion.py ==")
    autoprueba_extraccion()
    print("== Autoprueba fragmentacion.py ==")
    autoprueba_fragmentacion()


def test_documento(ruta: Path) -> None:
    """Corre extracción + fragmentación sobre ``ruta`` y valida el contrato de salida.

    No asume formato, idioma ni contenido: solo lo que ambos módulos
    garantizan para cualquier archivo soportado.
    """
    print(f"== Pipeline completo sobre {ruta.name} ==")
    assert ruta.is_file(), f"no existe: {ruta}"

    registros = extraer_documento(ruta)
    for r in registros:
        assert r["texto"].strip(), f"registro sin texto útil no debió llegar aquí: {r}"
        assert r["documento"] == ruta.name

    if not registros:
        print("OK - documento sin texto útil (0 registros, 0 fragmentos)")
        return

    fragmentos = fragmentar_registros(registros)
    assert fragmentos, "hay registros con texto pero no se generó ningún fragmento"

    for f in fragmentos:
        assert f["metadata"]["tokens"] <= MAX_TOKENS, (
            f"fragmento p{f['pagina']}/{f['fragmento']} excede el límite: "
            f"{f['metadata']['tokens']} > {MAX_TOKENS}"
        )
        assert f["texto"].strip()
        assert f["documento"] == ruta.name

    # ordinal de fragmento contiguo dentro de cada página/unidad
    por_pagina: dict[int, list[int]] = {}
    for f in fragmentos:
        por_pagina.setdefault(f["pagina"], []).append(f["fragmento"])
    for pagina, ordinales in por_pagina.items():
        assert ordinales == list(range(1, len(ordinales) + 1)), (pagina, ordinales)

    print(
        f"OK - {len(registros)} registros -> {len(fragmentos)} fragmentos, "
        f"max {max(f['metadata']['tokens'] for f in fragmentos)} tokens"
    )

    salida = ruta.parent / f"reporte_{ruta.stem}.md"
    generar_reporte(ruta, salida, fragmentos=fragmentos)


if __name__ == "__main__":
    test_etapas_aisladas()

    archivos = _archivos_a_probar()
    if not archivos:
        print(f"Sin archivos soportados en {CARPETA}. Formatos: {formatos_soportados()}")
    for ruta in archivos:
        test_documento(ruta)

    print("\nTODO OK")
