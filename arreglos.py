from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

paths = list(Path("/home/anothercoolcoder/pdfs").glob("*.pdf")) # directorio de archivos con la extension .pdf

resultados = {}

for result in converter.convert_all(paths):
    doc = result.document

    texto = doc.export_to_markdown()

    parrafos = [
        p.replace("\t", " ").strip() 
        for p in texto.split("\n\n") 
        if p.strip()
    ]
    tablas = [
        table.export_to_dataframe(doc).values.tolist()
        for table in doc.tables
    ]

    nombre_archivo = result.input.file.name

    resultados[nombre_archivo] = {
        "parrafos": parrafos,
        "tablas": tablas
    }

if resultados:
    primer_doc = next(iter(resultados))
    print(f"-- Párrafos de {primer_doc}")
    print(resultados[primer_doc]["parrafos"][:2])