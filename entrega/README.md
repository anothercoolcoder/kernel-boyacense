# Entregable CODEFEST AD ASTRA 2026

## Contenido

- `generador.py`: recuperacion BM25 + FAISS con CombSUM min-max y `alpha=0.1`.
- `base_vectorial/`: indice FAISS y metadata del encoder E5.
- `consultas.jsonl`: 50 consultas oficiales `q001` a `q050`.
- `requirements.txt`: dependencias de ejecucion en CPU.

## Instalacion

Usar Python 3.9.5 o superior en un entorno virtual:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

La primera ejecucion descarga `intfloat/multilingual-e5-large-instruct`
desde Hugging Face. Se requiere acceso de red durante esa descarga; despues
el modelo queda en la cache local de Hugging Face.

## Ejecucion

Los defaults se resuelven relativos a esta carpeta, no al directorio actual.
Puede ejecutarse desde cualquier directorio:

```bash
python /ruta/entrega/generador.py
```

Genera `/ruta/entrega/resultados.jsonl` con exactamente 50 registros.
Tambien admite rutas explicitas:

```bash
python generador.py \
  --consultas consultas.jsonl \
  --base-vectorial base_vectorial \
  --salida resultados.jsonl
```

## Contrato de salida

Cada consulta contiene exactamente 3 documentos y 10 fragmentos. Los
fragmentos pertenecen a los documentos devueltos, no duplican `chunk_id` y
contienen como maximo 250 palabras. La recuperacion no usa LLM generativo,
decoder ni reranker.
