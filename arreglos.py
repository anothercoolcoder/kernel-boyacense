from pathlib import Path
import pandas as pd
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

extensiones_soportadas = {"*.pdf", "*.html", "*.docx", "*.pptx"}
directorio = Path("/home/anothercoolcoder/files")

paths = [f for ext in extensiones_soportadas for f in directorio.glob(ext)]

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


resumen = []
parrafos_lista = []
tablas_lista = []

for archivo, contenido in resultados.items():
    resumen.append({
        "Archivo": archivo,
        "Total Párrafos": len(contenido["parrafos"]),
        "Total Tablas": len(contenido["tablas"])
    })
    
    for idx, p in enumerate(contenido["parrafos"], 1):
        parrafos_lista.append({
            "Archivo": archivo,
            "Índice Párrafo": idx,
            "Texto": p
        })
        
    for idx_t, tabla in enumerate(contenido["tablas"], 1):
        for idx_f, fila in enumerate(tabla, 1):
            parrafos_lista.append({
                "Archivo": archivo,
                "Tabla #": idx_t,
                "Fila #": idx_f,
                "Contenido Fila": " | ".join([str(c) for c in fila])
            })

with pd.ExcelWriter("comprobacion_docling.xlsx", engine="openpyxl") as writer:
    pd.DataFrame(resumen).to_excel(writer, sheet_name="Resumen", index=False)
    pd.DataFrame(parrafos_lista).to_excel(writer, sheet_name="Párrafos", index=False)
    if tablas_lista:
        pd.DataFrame(tablas_lista).to_excel(writer, sheet_name="Tablas", index=False)

print("¡Archivo comprobacion_docling.xlsx generado con éxito!")