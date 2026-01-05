# RAG FIA F1 Sporting Regulations 2025

## El problema

Este proyecto nace de una pregunta muy concreta: **¿cómo conseguir respuestas precisas cuando la información está dispersa en cientos de páginas de documentos técnicos complejos?**

Imagina buscar una regla específica en el reglamento deportivo de Fórmula 1 2025: más de 300 páginas donde una misma consulta puede requerir información de múltiples artículos, con referencias cruzadas, excepciones y casos especiales. **Leer el documento completo cada vez no es práctico. Confiar en la memoria de un modelo de IA puede llevar a respuestas inventadas.**

El desafío no era solo encontrar información, sino **garantizar que cada respuesta fuera verificable y trazable** hasta su fuente original en las regulaciones oficiales.

**La solución: RAG (Retrieval-Augmented Generation)**

En lugar de depender únicamente de lo que un modelo "recuerda" de su entrenamiento, RAG combina dos capacidades:

1. **Búsqueda inteligente**: Recupera solo los fragmentos relevantes de los documentos oficiales
2. **Generación contextual**: Utiliza esa información verificada para construir respuestas precisas

El resultado es un sistema que **nunca inventa**. Cada afirmación está respaldada por el texto original de las regulaciones, con referencias directas a los artículos correspondientes. Si la información no existe en los documentos, el sistema lo indica claramente en lugar de especular.

Este enfoque no solo resuelve el problema de la fiabilidad, sino que transforma 300+ páginas de regulaciones técnicas en un asistente conversacional que responde en segundos con información verificable y trazable.

---

## Los documentos utilizados

El sistema RAG se alimenta exclusivamente de documentación oficial.

- **PDF**: documento original, de la reglamentacion deportiva.
- **Markdown (.md)**: Su contenido son articulos densos con secciones y subsecciones.
- **Texto Plano (.txt)**: Su contenido son articulos menos densos, las secciones no contienen subsecciones.
- **CSV**: datos tabulares (puntos, clasificaciones, resultados).
- **JSON**: utilizado para realizar el Custionario de pregunta respuesta para realizar la evaluacion del modelo.
- **YAML**: utilizado para realizar el procesado de los datos en crudo que son los que fueron creados de forma manual.

---

## Preparación de los datos para el RAG

La preparación de los datos fue clave para el funcionamiento del sistema.

Primero, devido a la complegidad de automatizar la extraccion del contenido del pdf, a los formatos requeridos, se efectuo de forma manual, para ayudar al modelo se dividio por aticulos y apendices en documentos independientes, para facilitar que el contenido no estuviese mezclado. Cada fragmento contiene un conjunto de reglas relacionadas.

Después, se procesaron los ficheros en crudo, cambaindolos texto y agregando la informacion tabular a las secciones que lo requiriesen cuando se requiriese (esto es lo que se controla con el fichero previamente citado YAML).

---

## Elección del Modelo y Vector Database

### El Enfoque: Experimentación sobre Intuición

En lugar de elegir un modelo "porque sí", se diseño un experimento sistemático que evaluó **9 configuraciones diferentes**:

- 3 modelos de embeddings
- 3 estrategias de chunking (500, 1000, 2000 caracteres)
- Evaluación con 30+ queries reales

Cada configuración fue medida con múltiples métricas: F1 Score, NDCG, Precision@K, MAP, y MRR.

**Ventajas clave:**

- ✅ **Entrenado específicamente para retrieval**, no solo similitud semántica general
- ✅ **Balance perfecto**: 384 dimensiones capturan la semántica sin sobrecarga
- ✅ **Consistencia**: Destaca especialmente en queries difíciles con múltiples artículos
- ✅ **Eficiente**: 133MB, rápido, y funciona en CPU

**¿Por qué no usar modelos más grandes?**

1. El dominio es inglés técnico (su especialidad)
2. La optimización del chunking importa tanto como el modelo
3. Modelos pequeños bien optimizados > modelos grandes sin optimizar

### Estrategia de Chunking: 1000 caracteres con overlap de 200

**El experimento reveló un patrón claro:**

```
Chunk 500:   F1=0.798 → Pierde contexto
Chunk 1000:  F1=0.847 → Sweet spot ✓
Chunk 2000:  F1=0.812 → Demasiado genérico
```

**¿Por qué 1000 funcionó mejor?**

Las regulaciones F1 tienen una estructura natural:

- Párrafo principal (200-400 chars)
- Bullet points o sub-secciones (300-600 chars)
- Contexto adicional (100-200 chars)

**Total ≈ 800-1000 caracteres** por concepto completo.

Chunks de 500 partían conceptos a la mitad. Chunks de 2000 mezclaban conceptos no relacionados. **1000 caracteres captura exactamente un concepto completo** con su contexto.

El overlap de 200 asegura que no perdamos información en los "bordes" entre chunks.

### Vector Database: FAISS

La decisión fue práctica, no ideológica:

**Ventajas de FAISS:**

- ⚡ **Velocidad**: ~8ms por query
- 💰 **Costo**: $0 (local) vs. servicios de pago ($X/mes)
- 🔒 **Privacidad**: Datos completamente locales
- 📦 **Simplicidad**: No requiere infraestructura adicional obligatoria (Docker, servidores)
- 💾 **Eficiencia**: ~2.3 MB de índice para 1,500 chunks
- 🚀 **Deploy**: Funciona en cualquier servidor ? Ordenador con pocos recursos

---

## Evaluación

Tras realizar test manuales, se encontro que ciertas preguntas contenian mas contexto del deseado incluyendo en la respuesta información adicional que no esperaba.

a la pregunta de cuales eran los puntos y posiciones si no se finalizaba el 75% de la carrera, respondia correctamente, pero aportaba mas información que solo las posiciones y puntuaciones, como que minimo se tenia que disputar 2 vueltras en bandera verde.

En otras preguntas directamente daba respuestas inclorrectas, si que tenia el contexto pero era completamente erronea.

a la pregunta puntos y si la carrera finalizaba de forma completa, respondia que solo puntan los 8 primeros cuando en la realidad son 10, es posible que no entienda la pregunta ya que la carrera sprint si que son los 8 primeros y eso le confunda, por hacer preguntas con poco contexto.

En esta primera versión aun teniendo el mejor modelo del ensayo le falta mejorar.

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
