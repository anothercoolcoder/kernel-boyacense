from pathlib import Path
from docling.document_converter import DocumentConverter

source = "/home/anothercoolcoder/principito.pdf"  # archivo o enlace

converter = DocumentConverter()
result = converter.convert(source)
doc = result.document

parrafos = [item.text for item in doc.texts]

tablas = []

for table in doc.tables:
    df = table.export_to_dataframe()
    tablas.append(df.values.tolist())

print(f"Total de párrafos extraídos: {len(parrafos)}")
print("\nPrimeros 3 párrafos:")
print(parrafos[:3])

if tablas:
    print("\nPrimera tabla extraída como arreglo de filas:")
    print(tablas[0])