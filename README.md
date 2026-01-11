# RAG FIA F1 Sporting Regulations 2025

## El problema

Este proyecto nace de una pregunta muy concreta: **¿cómo conseguir respuestas precisas cuando la información está dispersa en cientos de páginas de documentos técnicos complejos?**

Imagina buscar una regla específica en el reglamento deportivo de Fórmula 1 2025: más de 300 páginas donde una misma consulta puede requerir información de múltiples artículos, con referencias cruzadas, excepciones y casos especiales. **Leer el documento completo cada vez no es práctico. Confiar en la memoria de un modelo de IA puede llevar a respuestas inventadas.**

El desafío no era solo encontrar información, sino **garantizar que cada respuesta fuera verificable y trazable** hasta su fuente original en las regulaciones oficiales.

---

## La solución: RAG (Retrieval-Augmented Generation)

En lugar de depender únicamente de lo que un modelo "recuerda" de su entrenamiento, RAG combina dos capacidades:

1. **Búsqueda inteligente**: Recupera solo los fragmentos relevantes de los documentos oficiales
2. **Generación contextual**: Utiliza esa información verificada para construir respuestas precisas

El resultado es un sistema que **nunca inventa**. Cada afirmación está respaldada por el texto original de las regulaciones, con referencias directas a los artículos correspondientes. Si la información no existe en los documentos, el sistema lo indica claramente en lugar de especular.

Este enfoque no solo resuelve el problema de la fiabilidad, sino que transforma 300+ páginas de regulaciones técnicas en un asistente conversacional que responde en segundos con información verificable y trazable.

---

## Los documentos utilizados

El sistema RAG trabaja con distintos formatos documentales a lo largo de su pipeline.

No todos los formatos cumplen el mismo rol dentro del sistema: algunos forman parte de la base de conocimiento indexada y recuperable por el asistente, mientras que otros se utilizan como soporte para el procesado, estructuración y evaluación del sistema.

- **PDF**: documento original de la reglamentación deportiva. Forma parte del conocimiento base del asistente.
- **Markdown (.md)**: artículos densos con secciones y subsecciones, utilizados como conocimiento estructurado para el RAG.
- **Texto plano (.txt)**: artículos menos densos sin subsecciones, utilizados como conocimiento complementario.
- **CSV**: datos tabulares (puntos, clasificaciones, resultados) empleados como información estructurada de apoyo.
- **JSON**: utilizado para construir el cuestionario de preguntas y respuestas empleado en la evaluación del sistema.
- **YAML**: utilizado en el procesado de los datos en crudo, creados manualmente durante la fase de preparación de la información.

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

A la pregunta de cuales eran los puntos y posiciones si no se finalizaba el 75% de la carrera, respondia correctamente, pero aportaba mas información que solo las posiciones y puntuaciones, como que minimo se tenia que disputar 2 vueltras en bandera verde.

En otras preguntas directamente daba respuestas inclorrectas, si que tenia el contexto pero era completamente erronea.

A la pregunta puntos y si la carrera finalizaba de forma completa, respondia que solo puntan los 8 primeros cuando en la realidad son 10, es posible que no entienda la pregunta ya que la carrera sprint si que son los 8 primeros y eso le confunda, por hacer preguntas con poco contexto.

En esta primera versión aun teniendo el mejor modelo del ensayo le falta mejorar.

---

## Mejoras

Como parte del análisis de posibles mejoras del sistema RAG, se exploraron distintas estrategias orientadas a incrementar la precisión y coherencia de las respuestas generadas, así como a reducir errores derivados de la recuperación de contexto.

Una de las aproximaciones evaluadas consistió en modificar la estrategia de preprocesado documental, dividiendo cada artículo en secciones independientes antes de su indexación. La hipótesis inicial era que una segmentación más fina permitiría recuperar fragmentos más específicos y directamente relacionados con la consulta del usuario.

Sin embargo, los resultados obtenidos mostraron respuestas más incompletas y con menor coherencia global, debido principalmente a la pérdida de contexto entre secciones relacionadas del mismo artículo. Esta fragmentación excesiva dificultó la reconstrucción del significado completo de ciertas normas, por lo que la estrategia fue descartada tras su evaluación, manteniéndose una segmentación que preserva mayor continuidad semántica entre fragmentos.

Como complemento al sistema base, se incorporó una fase de postprocesado de las respuestas orientada a mejorar su legibilidad y naturalidad, sin alterar el contenido factual recuperado. Esta mejora tiene un impacto principalmente en la experiencia de uso del asistente, pero no modifica el proceso de recuperación ni la exactitud de la información proporcionada.

Finalmente, se identifican como líneas futuras de mejora la especialización del sistema en múltiples modelos orientados a distintas categorías normativas, aprovechando la clasificación existente en los ficheros YAML. Esta aproximación permitiría reducir el espacio de búsqueda de cada modelo y, mediante un mecanismo de orquestación de intenciones, construir un sistema más escalable y potencialmente más preciso, donde cada componente se especialice en un subconjunto concreto de la normativa.

---

## Conlusiones

La primera lección extraída de este proyecto es que un sistema RAG y un modelo LLM no representan la misma arquitectura de interpretación del lenguaje natural, aunque pueden complementarse. Mientras que un LLM opera principalmente sobre el conocimiento implícito adquirido durante su entrenamiento, un sistema RAG fundamenta sus respuestas en información externa recuperada dinámicamente, lo que resulta clave en contextos donde la trazabilidad y la fidelidad a las fuentes son requisitos esenciales.

Otro aspecto fundamental identificado es la importancia del procesado y la estructuración previa de la información incorporada al sistema RAG. La forma en que se agrupa o segmenta el contenido antes de su indexación tiene un impacto directo en la calidad de las respuestas. En particular, parámetros como el tamaño de los fragmentos de texto y el grado de solapamiento entre ellos resultan determinantes para lograr un equilibrio adecuado entre cobertura contextual y precisión semántica, como se ha observado al evaluar distintas estrategias de segmentación documental.

Asimismo, el modelo base empleado actúa como traductor de intenciones entre la consulta del usuario y el conocimiento recuperado. La elección de este modelo condiciona aspectos como el idioma de entrada y la correcta interpretación de las preguntas planteadas. Si el modelo no está alineado con el lenguaje utilizado en las consultas, la recuperación de información y la posterior construcción de respuestas se ven inevitablemente afectadas, independientemente de la calidad del sistema RAG subyacente.

En relación con el prompting, el proyecto ha puesto de manifiesto que los ajustes realizados a nivel del prompt conversacional del chatbot tienen un impacto limitado sobre la precisión factual de las respuestas en un sistema RAG. Este resultado refuerza la idea de que, en arquitecturas basadas en recuperación de contexto, la calidad del sistema depende principalmente de la fase de recuperación, del preprocesado de los documentos y de los criterios utilizados para seleccionar y presentar la evidencia, más que del estilo conversacional del asistente.

Finalmente, se observa que concentrar grandes volúmenes de información heterogénea en un único modelo puede incrementar el riesgo de respuestas ambiguas o incompletas cuando existen contenidos similares. Como línea futura de mejora, la segmentación del conocimiento en modelos especializados por categoría normativa, junto con un mecanismo de orquestación de intenciones, se presenta como una estrategia prometedora para reducir errores y mejorar la precisión global del sistema.

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

# Modo Compare: dada una consulta debuelve el resultado sin tratat y tratado con el system prompt y algo mas de codigo
python3 ./script/test_chatbot.py --mode compare
```

Ejecuta la interfaz de usuario grafica creada con streamlit

```bash
streamlit run ./scripts/app_chatbot.py
```
