# RAG FIA F1 Sporting Regulations 2025

---

## Inicialización y Uso

Proyecto realizado con la versión 3.11 de python, no se puede garantizar que con versiones inferiores funcione todas la librerias y garantizado que versiones posteriores librerias LangChain aun no son compatibles

```bash
# Crear entorno virtual con version especifica
python3.11 -m venv .venv

# Activar entono virtual en entorno Unix (Linux y Mac)
source ./.venv/bin/activate

# Instalar librerias necesarias
pip3 install -r requirements.txt
```

Procesar documentos crudos para crear el rag

```bash
python3 ./script/process_raw_documents.py
```

Creacion de los modelos RAG y evalueacion del mejor

```bash
python3 ./script/evaluate_rag_strategies.py
```

Ejecución del chatbot en linea de comandos para test

```bash
# Modo Test: carga el mejor modelo guardado y ejecuta una consulta predefinida
python3 ./script/test_chatbot.py --mode test

# Modo Multiple: carga el mejor modelo guardado y ejecuta tres consultas predefinidas
python3 ./script/test_chatbot.py --mode multiple

# Modo Interactive: carga el mejor modelo guardado y ejecuta tipo consulta respuesta, es decir la consulta se puede hacer de forma personalizada
python3 ./script/test_chatbot.py --mode interactive
```

Ejecuta la interfaz de usuario grafica creada con streamlit

```bash
streamlit run ./scripts/app_chatbot.py
```
