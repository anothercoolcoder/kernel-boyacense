"""Suite de pruebas end-to-end del pipeline RAG — kernel-boyacense.

Ejecuta las autopruebas unitarias de cada módulo y luego corre el pipeline
completo (extracción → fragmentación) sobre archivos reales de ``corpus_adl/``
o los pasados por argumento CLI.

Uso:
    python test_pipeline.py                  # prueba todo archivo soportado en corpus_adl/
    python test_pipeline.py a.pdf b.png ...   # prueba solo los archivos dados

Nota: la indexación FAISS (etapa 3) NO se ejecuta aquí para mantener los
tests rápidos. Para validar el pipeline completo usa:
    python main.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows es cp1252

# ── Imports del pipeline ──────────────────────────────────────────────────── #
from extraccion.extraccion import (
    _autoprueba as autoprueba_extraccion,
    extraer_documento,
    formatos_soportados,
    ErrorExtraccion,
)
from extraccion.fragmentacion import (
    MAX_TOKENS,
    _autoprueba as autoprueba_fragmentacion,
    fragmentar_registros,
)
from extraccion.reporte_chunks import generar_reporte

# ── Ruta de corpus por defecto ────────────────────────────────────────────── #
BASE_DIR   = Path(__file__).parent
CORPUS_ADL = BASE_DIR / "corpus_adl"


def _archivos_a_probar() -> list[Path]:
    """Archivos pasados por CLI, o todo archivo soportado en corpus_adl/."""
    if len(sys.argv) > 1:
        return [Path(a) for a in sys.argv[1:]]
    soportadas = set(formatos_soportados())
    return sorted(
        p
        for p in CORPUS_ADL.rglob("*")
        if p.is_file()
        and p.suffix.lower() in soportadas
        and not p.name.startswith("reporte_")
    )


def test_etapas_aisladas() -> None:
    print("== Autoprueba extraccion.py ==")
    autoprueba_extraccion()
    print("== Autoprueba fragmentacion.py ==")
    autoprueba_fragmentacion()


def test_documento(ruta: Path) -> None:
    """Corre extracción + fragmentación sobre ``ruta`` y valida el contrato de salida."""
    print(f"== Pipeline completo sobre {ruta.name} ==")
    assert ruta.is_file(), f"no existe: {ruta}"

    try:
        registros = extraer_documento(ruta)
    except ErrorExtraccion as exc:
        print(f"OK - archivo ignorado por ErrorExtraccion controlado: {exc}")
        return

    for r in registros:
        assert r["texto"].strip(), f"registro sin texto útil no debió llegar aquí: {r}"
        assert r["documento"] == ruta.name

    if not registros:
        print("OK - documento sin texto útil (0 registros, 0 fragmentos)")
        return

    fragmentos = fragmentar_registros(registros)
    assert fragmentos, "hay registros con texto pero no se generó ningún fragmento"

    for f in fragmentos:
        assert f["num_tokens"] <= MAX_TOKENS, (
            f"fragmento p{f['_meta'].get('pagina')}/{f['posicion']} excede el límite: "
            f"{f['num_tokens']} > {MAX_TOKENS}"
        )
        assert f["texto"].strip()
        assert Path(f["fuente"]).name == ruta.name

    # Ordinal de posicion continuo y monotonicamente creciente para todo el documento
    posiciones = [f["posicion"] for f in fragmentos]
    assert posiciones == list(range(len(fragmentos))), f"posicion no es monotónica: {posiciones}"

    # chunk_id debe ser único por documento
    chunk_ids = [f["chunk_id"] for f in fragmentos]
    assert len(chunk_ids) == len(set(chunk_ids)), f"chunk_id duplicados encontrados: {chunk_ids}"

    # Validar metadatos obligatorios (§3.4)
    for f in fragmentos:
        assert f["fenomeno"] in (1, 2, 3), f"fenomeno no válido o None: {f['fenomeno']}"
        assert f["formato"] and "/" not in f["formato"], f"formato no válido: {f['formato']}"
        assert f["fuente"] and not f["fuente"].startswith("/"), f"fuente no debe ser ruta absoluta: {f['fuente']}"

    print(
        f"OK - {len(registros)} registros → {len(fragmentos)} fragmentos, "
        f"max {max(f['num_tokens'] for f in fragmentos)} tokens"
    )

    salida = BASE_DIR / f"reporte_{ruta.stem}.md"
    generar_reporte(ruta, salida, fragmentos=fragmentos)


if __name__ == "__main__":
    test_etapas_aisladas()

    archivos = _archivos_a_probar()
    if not archivos:
        print(f"Sin archivos soportados en {CORPUS_ADL}. Formatos: {formatos_soportados()}")
    for ruta in archivos:
        test_documento(ruta)

    print("\nTODO OK")
