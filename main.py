import argparse
import json
from math import ceil
from pathlib import Path
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from PIL import Image
import easyocr

try:
    from pyrosm import OSM
except ImportError:
    OSM = None

# --------------------------------------------------
# 1. CONFIGURACIÓN DE MODELOS
# --------------------------------------------------
# Segmentador sintáctico liviano con spaCy
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Modelo de embeddings para la medición semántica entre oraciones
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convertidor de Docling
converter = DocumentConverter()

ocr_reader = easyocr.Reader(
    ["es", "en", "pt"],
    gpu=True,
)


# --------------------------------------------------
# 2. FUNCIÓN DE CHUNKING SEMÁNTICO
# --------------------------------------------------
def generar_chunks_semanticos(
    oraciones: list[str], threshold: float = 0.65
) -> list[dict]:
    if not oraciones:
        return []

    # 1. Generar vectores para cada oración
    embeddings = embedder.encode(oraciones, normalize_embeddings=True)

    chunks = []
    chunk_actual = [oraciones[0]]

    for i in range(len(oraciones) - 1):
        vec_actual = embeddings[i]
        vec_siguiente = embeddings[i + 1]

        # Similitud Coseno (al estar normalizados, es el producto punto)
        similitud = float(np.dot(vec_actual, vec_siguiente))

        if similitud >= threshold:
            # Mantienen la misma línea temática
            chunk_actual.append(oraciones[i + 1])
        else:
            # Cambio abrupto de tema -> Cerrar chunk anterior y comenzar uno nuevo
            chunks.append(
                {
                    "content": " ".join(chunk_actual),
                    "sentence_count": len(chunk_actual),
                }
            )
            chunk_actual = [oraciones[i + 1]]

    # Agregar el último chunk resultante
    if chunk_actual:
        chunks.append(
            {
                "content": " ".join(chunk_actual),
                "sentence_count": len(chunk_actual),
            }
        )

    return chunks


def extraer_texto_ocr(imagen) -> str:
    """
    Ejecuta EasyOCR sobre una imagen.

    Acepta:
      - una ruta/str a un archivo de imagen en disco, o
      - un objeto PIL.Image.Image ya cargado en memoria (por ejemplo,
        el que devuelve Docling para cada figura extraída del PDF).
    """
    try:
        # EasyOCR necesita un array numpy o una ruta de archivo; si nos
        # llega un PIL.Image lo convertimos primero.
        if isinstance(imagen, Image.Image):
            entrada = np.array(imagen.convert("RGB"))
        else:
            entrada = imagen  # ya es una ruta (str/Path)

        texto = ocr_reader.readtext(
            entrada,
            detail=0,
            paragraph=True,
        )

        return "\n".join(texto).strip()

    except Exception as e:
        print(f"  Error en OCR: {e}")
        return ""


# --------------------------------------------------
# 3. PROCESAMIENTO Y PARSEO DEL PDF
# --------------------------------------------------
def procesar_pdf(pdf_path: str) -> dict:
    result = converter.convert(pdf_path)
    doc = result.document

    salida = {
        "paragraphs": [],
        "semantic_chunks": [],
        "tables": [],
        "images": [],
        "equations": [],
    }

    todas_las_oraciones = []

    # --- A. PARRAFO Y EXTRACCIÓN DE ORACIONES ---
    print("\n====================")
    print("PÁRRAFOS Y ORACIONES")
    print("====================")

    contador_oracion = 1
    for text_node in doc.texts:
        contenido = text_node.text.strip()
        if not contenido:
            continue

        # Segmentación sintáctica previa con spaCy
        block = nlp(contenido)
        sents = [
            sent.text.strip() for sent in block.sents if sent.text.strip()
        ]

        for frase in sents:
            print(f"[{contador_oracion}] {frase}")
            todas_las_oraciones.append(frase)
            contador_oracion += 1

        salida["paragraphs"].append(
            {
                "label": getattr(text_node, "label", "text"),
                "text": contenido,
                "sentences": sents,
            }
        )

    # --- B. GENERACIÓN DE CHUNKS SEMÁNTICOS ---
    print("\n====================")
    print("GENERANDO CHUNKS SEMÁNTICOS")
    print("====================")

    semantic_chunks = generar_chunks_semanticos(
        todas_las_oraciones, threshold=0.65
    )
    salida["semantic_chunks"] = semantic_chunks

    for idx, chk in enumerate(semantic_chunks, 1):
        print(f"\n--- Chunk Semántico {idx} ({chk['sentence_count']} oraciones) ---")
        print(chk["content"])

    # --- C. TABLAS ---
    print("\n====================")
    print("TABLAS")
    print("====================")

    for i, table in enumerate(doc.tables):
        try:
            # doc=doc preserva la grilla de filas y columnas en pandas
            df = table.export_to_dataframe(doc=doc)

            print(f"\n--- Tabla {i+1} ---")
            print(df.to_string())

            salida["tables"].append(
                {"table_index": i + 1, "data": df.to_dict(orient="records")}
            )
        except Exception as e:
            print(f"Error procesando Tabla {i+1}: {e}")

    # --- D. FIGURAS (con OCR e interpretación de diagramas) ---
    if hasattr(doc, "pictures") and doc.pictures:
        print("\n====================")
        print("FIGURAS (CON INTERPRETACIÓN DE DIAGRAMAS)")
        print("====================")

        for i, pic in enumerate(doc.pictures):
            print(f"\nFigura {i+1} detectada.")
            meta_data = getattr(pic, "meta", getattr(pic, "annotations", {}))

            texto_ocr = ""
            interpretacion_diagrama = {
                "estados": [],
                "transiciones": [],
                "etiquetas": [],
                "componentes_identificados": []
            }

            try:
                # Extraer imagen
                imagen_pil = pic.get_image(doc)

                if imagen_pil is not None:
                    # OCR con detail=1 para obtener confianza y posición
                    resultado_detallado = ocr_reader.readtext(
                        np.array(imagen_pil.convert("RGB")),
                        detail=1,
                    )

                    # Extraer texto completo
                    lineas_ocr = []
                    for bbox, texto, confianza in resultado_detallado:
                        if confianza > 0.3:  # Umbral de confianza
                            lineas_ocr.append(texto)

                    texto_ocr = "\n".join(lineas_ocr)

                    if texto_ocr:
                        print(f"  OCR Figura {i+1} ({len(lineas_ocr)} elementos detectados):")

                        # Análisis heurístico del contenido OCR para identificar patrones
                        for linea in lineas_ocr:
                            linea_clean = linea.strip()

                            # Detectar estados (típicamente: q0, q1, s0, S_even, etc.)
                            if any(x in linea_clean.lower() for x in ['q', 's', 'state']) and \
                               (linea_clean[0].lower() in ['q', 's'] or
                                'state' in linea_clean.lower()):
                                interpretacion_diagrama["estados"].append(linea_clean)

                            # Detectar transiciones (contienen entrada/salida, flechas, etc.)
                            if any(x in linea_clean for x in ['/', '0', '1', '→', '->', 'input', 'output']):
                                interpretacion_diagrama["transiciones"].append(linea_clean)

                            # Etiquetas generales
                            interpretacion_diagrama["etiquetas"].append(linea_clean)

                        # Identificar tipo de diagrama
                        texto_lower = texto_ocr.lower()
                        if 'state' in texto_lower or 'q0' in texto_lower or 'q1' in texto_lower:
                            interpretacion_diagrama["componentes_identificados"].append(
                                "Diagrama de máquina de estados (FSM/FST)"
                            )
                        if 'parity' in texto_lower or 'detector' in texto_lower:
                            interpretacion_diagrama["componentes_identificados"].append(
                                "Máquina especializada (verificador de paridad/detector)"
                            )
                        if any(x in texto_lower for x in ['mealy', 'moore']):
                            tipo = "Mealy" if "mealy" in texto_lower else "Moore"
                            interpretacion_diagrama["componentes_identificados"].append(
                                f"Máquina de {tipo}"
                            )

                        print(f"    Texto extraído:\n{texto_ocr[:300]}{'...' if len(texto_ocr) > 300 else ''}")
                        if interpretacion_diagrama["componentes_identificados"]:
                            print(f"    Identificado como: {', '.join(interpretacion_diagrama['componentes_identificados'])}")
                    else:
                        print(f"  OCR Figura {i+1}: sin texto detectado.")
                else:
                    print(f"  Figura {i+1} no tiene imagen extraíble.")

            except Exception as e:
                print(f"  Error extrayendo imagen de la Figura {i+1}: {e}")

            salida["images"].append(
                {
                    "figure_index": i + 1,
                    "meta": str(meta_data),
                    "ocr_text": texto_ocr,
                    "diagram_interpretation": interpretacion_diagrama,
                }
            )

    # --- E. ECUACIONES (con OCR si son imágenes) ---
    print("\n====================")
    print("ECUACIONES")
    print("====================")

    ecuaciones_encontradas = []

    # 1. Buscar nodos de texto etiquetados como fórmulas/ecuaciones
    for text_node in doc.texts:
        label = str(getattr(text_node, "label", "")).lower()
        if label in ["formula", "equation", "math"]:
            eq_text = text_node.text.strip()
            if eq_text:
                print(f"Ecuación (etiqueta): {eq_text}")
                ecuaciones_encontradas.append({
                    "type": "text_node",
                    "content": eq_text,
                    "source": "labeled_text_node"
                })

    # 2. Si hay imágenes que podrían ser ecuaciones/fórmulas,
    #    aplicar OCR adicional para capturar fórmulas en ellas
    if hasattr(doc, "pictures") and doc.pictures:
        for i, pic in enumerate(doc.pictures):
            try:
                imagen_pil = pic.get_image(doc)
                if imagen_pil is not None:
                    # Aplicar OCR con modo "detail=1" para obtener confianza
                    # y poder filtrar mejor el contenido matemático
                    resultado_ocr = ocr_reader.readtext(
                        np.array(imagen_pil.convert("RGB")),
                        detail=1,  # Retorna [bbox, texto, confianza]
                    )

                    # Extraer texto de fórmulas/símbolos matemáticos
                    formula_text = "\n".join([
                        item[1] for item in resultado_ocr
                        if item[2] > 0.3  # Filtrar por confianza mínima
                    ])

                    if formula_text.strip():
                        # Detectar si es probablemente una fórmula/ecuación
                        # (contiene símbolos matemáticos típicos)
                        simbolos_math = ['=', '+', '-', '×', '÷', '∑', '∏',
                                        'λ', 'δ', 'Σ', 'Γ', '→', '×', '→']
                        es_formula = any(s in formula_text for s in simbolos_math)

                        if es_formula:
                            print(f"Ecuación/Fórmula (imagen {i+1}): {formula_text}")
                            ecuaciones_encontradas.append({
                                "type": "image",
                                "content": formula_text,
                                "source": f"picture_{i+1}",
                                "confidence": "mixed"
                            })

            except Exception as e:
                print(f"  Nota: No se pudo extraer fórmula adicional de imagen {i+1}: {e}")

    salida["equations"] = ecuaciones_encontradas

    return salida


# --------------------------------------------------
# 3-BIS. PROCESAMIENTO Y PARSEO DE ARCHIVOS .PBF (OpenStreetMap) CON PYROSM
# --------------------------------------------------
def procesar_pbf(pbf_path: str, chunk_size: int = 200) -> dict:
    """
    Procesa un archivo .pbf (extracto de OpenStreetMap) usando la librería
    Pyrosm.

    A diferencia de un PDF (que se parte en oraciones/párrafos), un .pbf
    contiene entidades geoespaciales (edificios, vías, POIs, límites
    administrativos). Aquí el equivalente a "párrafo" es un FRAGMENTO:
    un lote (CHUNK) de `chunk_size` entidades de una misma capa, con una
    descripción textual breve que luego se puede reutilizar para
    embeddings/LLMs igual que con los chunks semánticos de texto.
    """
    if OSM is None:
        raise ImportError(
            "La librería 'pyrosm' no está instalada. Instálala con: "
            "pip install pyrosm"
        )

    osm = OSM(pbf_path)

    salida = {
        "paragraphs": [],
        "semantic_chunks": [],
        "tables": [],
        "images": [],
        "equations": [],
        "layers_summary": {},
        "fragments": [],
    }

    # --- A. EXTRAER LAS CAPAS DISPONIBLES DEL .PBF ---
    capas_extractoras = {
        "buildings": lambda: osm.get_buildings(),
        "network_driving": lambda: osm.get_network(network_type="driving"),
        "pois": lambda: osm.get_pois(),
        "boundaries": lambda: osm.get_boundaries(),
    }

    capas_datos = {}
    for nombre_capa, extractor in capas_extractoras.items():
        try:
            gdf = extractor()
        except Exception as e:
            print(f"  Nota: no se pudo extraer la capa '{nombre_capa}': {e}")
            gdf = None
        capas_datos[nombre_capa] = gdf

    # --- B. RESUMEN AL COMIENZO: cuántos fragmentos/chunks se generarán ---
    total_features = 0
    total_chunks_previstos = 0

    for nombre_capa, gdf in capas_datos.items():
        n_features = 0 if gdf is None or gdf.empty else len(gdf)
        n_chunks = ceil(n_features / chunk_size) if n_features else 0
        salida["layers_summary"][nombre_capa] = {
            "features": n_features,
            "chunks_previstos": n_chunks,
        }
        total_features += n_features
        total_chunks_previstos += n_chunks

    print("\n====================")
    print(f"RESUMEN INICIAL DE EXTRACCIÓN PBF: {Path(pbf_path).name}")
    print("====================")
    print(f"Entidades OSM encontradas   : {total_features}")
    print(f"Fragmentos/chunks previstos : {total_chunks_previstos} "
          f"(tamaño de chunk = {chunk_size} entidades)")
    for nombre_capa, info in salida["layers_summary"].items():
        print(f"  - {nombre_capa:<16}: {info['features']:>7} entidades -> "
              f"{info['chunks_previstos']:>4} chunks")
    print("====================\n")

    # --- C. GENERAR FRAGMENTOS (CHUNKS) POR CAPA ---
    print("====================")
    print("FRAGMENTOS Y CHUNKS POR CAPA")
    print("====================")

    contador_fragmento = 1
    for nombre_capa, gdf in capas_datos.items():
        if gdf is None or gdf.empty:
            continue

        print(f"\n--- Capa: {nombre_capa} ({len(gdf)} entidades) ---")

        columnas_relevantes = [c for c in gdf.columns if c != "geometry"][:8]

        for inicio in range(0, len(gdf), chunk_size):
            lote = gdf.iloc[inicio:inicio + chunk_size]
            indice_chunk = inicio // chunk_size + 1

            # Un par de ejemplos legibles para describir el chunk
            ejemplos = []
            for _, fila in lote.head(3).iterrows():
                nombre = fila.get("name") if "name" in lote.columns else None
                descriptor = (
                    nombre if isinstance(nombre, str) and nombre.strip()
                    else str(fila.get("id", "sin-id"))
                )
                ejemplos.append(descriptor)

            contenido_texto = (
                f"Capa '{nombre_capa}', chunk {indice_chunk}: "
                f"{len(lote)} entidades. Ejemplos: "
                f"{', '.join(ejemplos) if ejemplos else 'N/A'}."
            )

            try:
                datos_lote = json.loads(
                    lote[columnas_relevantes].to_json(orient="records")
                ) if columnas_relevantes else []
            except Exception as e:
                print(f"    Nota: no se pudo serializar el chunk {indice_chunk}: {e}")
                datos_lote = []

            fragmento = {
                "fragment_index": contador_fragmento,
                "layer": nombre_capa,
                "chunk_index": indice_chunk,
                "feature_count": len(lote),
                "columns": columnas_relevantes,
                "content": contenido_texto,
                "data": datos_lote,
            }

            salida["fragments"].append(fragmento)
            print(f"  [Fragmento {contador_fragmento}] {contenido_texto}")
            contador_fragmento += 1

    # --- D. CHUNKS SEMÁNTICOS SOBRE LAS DESCRIPCIONES DE LOS FRAGMENTOS ---
    print("\n====================")
    print("GENERANDO CHUNKS SEMÁNTICOS (sobre descripciones de fragmentos OSM)")
    print("====================")

    descripciones = [f["content"] for f in salida["fragments"]]
    semantic_chunks = generar_chunks_semanticos(descripciones, threshold=0.65)
    salida["semantic_chunks"] = semantic_chunks

    for idx, chk in enumerate(semantic_chunks, 1):
        print(f"\n--- Chunk Semántico {idx} ({chk['sentence_count']} fragmentos agrupados) ---")
        print(chk["content"])

    return salida


# --------------------------------------------------
# 4. UTILIDADES PARA MANEJAR RUTAS (ARCHIVO O CARPETA)
# --------------------------------------------------
# Docling detecta el formato automáticamente según la extensión del archivo,
# así que el mismo pipeline de extracción (párrafos, tablas, figuras,
# ecuaciones) funciona para todos estos tipos de entrada.
EXTENSIONES_SOPORTADAS = {
    ".pdf",
    ".json",   # Docling JSON (documento ya exportado por Docling)
    ".md",
    ".xlsx",
    ".html",
    ".htm",
    ".csv",
    ".pbf",    # extractos de OpenStreetMap, procesados con Pyrosm (no con Docling)
}


def resolver_lista_archivos(ruta: Path, recursivo: bool = False) -> list[Path]:
    """
    Dada una ruta que puede ser:
      - un archivo individual (pdf, json, md, xlsx, html, csv, ...), o
      - una carpeta que contiene uno o más de esos archivos,
    devuelve la lista de rutas a procesar.
    """
    if ruta.is_file():
        if ruta.suffix.lower() not in EXTENSIONES_SOPORTADAS:
            raise ValueError(
                f"El archivo '{ruta}' tiene una extensión no soportada "
                f"({ruta.suffix}). Soportadas: {', '.join(sorted(EXTENSIONES_SOPORTADAS))}"
            )
        return [ruta]

    if ruta.is_dir():
        archivos = []
        for ext in EXTENSIONES_SOPORTADAS:
            patron = f"**/*{ext}" if recursivo else f"*{ext}"
            archivos.extend(ruta.glob(patron))

        archivos = sorted(set(archivos))

        if not archivos:
            raise FileNotFoundError(
                f"No se encontraron archivos soportados en la carpeta '{ruta}'"
                + (" (búsqueda recursiva)" if recursivo else "")
                + f". Extensiones buscadas: {', '.join(sorted(EXTENSIONES_SOPORTADAS))}"
            )
        return archivos

    raise FileNotFoundError(f"La ruta '{ruta}' no existe.")


def procesar_y_guardar(pdf_path: Path, carpeta_salida: Path, chunk_size: int = 200) -> None:
    """Procesa un único archivo y guarda su JSON correspondiente.

    Los .pbf (OpenStreetMap) se procesan con Pyrosm; el resto de formatos
    soportados (.pdf, .json, .md, .xlsx, .html, .csv) se procesan con Docling.
    """
    print("\n" + "#" * 60)
    print(f"# PROCESANDO: {pdf_path}")
    print("#" * 60)

    if pdf_path.suffix.lower() == ".pbf":
        datos = procesar_pbf(str(pdf_path), chunk_size=chunk_size)
    else:
        datos = procesar_pdf(str(pdf_path))

    if isinstance(datos, dict) and any(datos.values()):
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        output_json = carpeta_salida / f"{pdf_path.stem}_parsed.json"

        with open(output_json, "w", encoding="utf-16") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

        print("\n===================================")
        print(f"RESUMEN DE EXTRACCIÓN: {pdf_path.name}")
        print("===================================")
        print(f"Párrafos          : {len(datos['paragraphs'])}")
        print(f"Chunks Semánticos : {len(datos['semantic_chunks'])}")
        print(f"Tablas            : {len(datos['tables'])}")
        print(f"Figuras (diagr.)  : {len(datos['images'])}")
        print(f"Ecuaciones        : {len(datos['equations'])}")

        if datos.get('layers_summary'):
            print(f"Fragmentos OSM    : {len(datos.get('fragments', []))}")
            print(f"\n  Capas OSM procesadas:")
            for capa, info in datos['layers_summary'].items():
                print(f"    - {capa:<16}: {info['features']:>7} entidades en {info['chunks_previstos']:>4} chunks")

        if datos['equations']:
            print(f"\n  Ecuaciones detectadas:")
            for j, eq in enumerate(datos['equations'], 1):
                fuente = eq.get('source', 'desconocida')
                print(f"    {j}. [{fuente}] {eq['content'][:80]}{'...' if len(eq['content']) > 80 else ''}")

        if datos['images']:
            print(f"\n  Diagramas interpretados:")
            for j, img in enumerate(datos['images'], 1):
                diag = img.get('diagram_interpretation', {})
                n_elementos = len(diag.get('etiquetas', []))
                componentes = ', '.join(diag.get('componentes_identificados', ['sin clasificar']))
                print(f"    Fig. {img['figure_index']}: {n_elementos} elementos | {componentes}")

        print(f"\n  Archivo guardado en: '{output_json}'")
    else:
        print(f"  No se pudieron extraer datos del PDF '{pdf_path.name}'.")


# --------------------------------------------------
# 5. EJECUCIÓN PRINCIPAL (ARCHIVO O CARPETA)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Procesa uno o varios documentos (pdf, json, md, xlsx, html, csv) "
                    "-- chunking semántico, tablas, figuras con OCR y ecuaciones -- "
                    "y exporta el resultado a JSON."
    )
    parser.add_argument(
        "ruta",
        nargs="?",
        default="taller.pdf",
        help="Ruta a un archivo individual o a una carpeta que contenga varios "
             "documentos soportados (.pdf, .json, .md, .xlsx, .html, .csv). "
             "Por defecto: 'taller.pdf'",
    )
    parser.add_argument(
        "-o", "--output",
        default="salida_json",
        help="Carpeta donde se guardarán los JSON generados "
             "(solo aplica cuando la ruta es una carpeta o "
             "se quiere forzar el directorio de salida). Default: 'salida_json'",
    )
    parser.add_argument(
        "-r", "--recursivo",
        action="store_true",
        help="Si la ruta es una carpeta, buscar documentos también en subcarpetas.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Tamaño de lote (número de entidades OSM) por fragmento/chunk "
             "al procesar archivos .pbf. Default: 200",
    )

    args = parser.parse_args()
    ruta = Path(args.ruta)
    carpeta_salida = Path(args.output)

    try:
        archivos = resolver_lista_archivos(ruta, recursivo=args.recursivo)
    except (FileNotFoundError, ValueError) as e:
        print(f" {e}")
        return

    print(f"Se encontraron {len(archivos)} archivo(s) para procesar.")

    errores = []
    for archivo_path in archivos:
        try:
            procesar_y_guardar(archivo_path, carpeta_salida, chunk_size=args.chunk_size)
        except Exception as e:
            print(f" Error procesando '{archivo_path.name}': {e}")
            errores.append((archivo_path.name, str(e)))

    print("\n" + "=" * 60)
    print("RESUMEN GLOBAL")
    print("=" * 60)
    print(f"Total procesados intentados : {len(archivos)}")
    print(f"Con errores                 : {len(errores)}")
    if errores:
        for nombre, err in errores:
            print(f"  - {nombre}: {err}")


if __name__ == "__main__":
    main()