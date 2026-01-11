import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

class F1RAGChatbot:
    """
    Chatbot RAG para consultas sobre regulaciones F1 2025
    
    Características:
    - Recuperación de documentos con reranking
    - Filtrado inteligente de contexto
    - Extracción de información clave
    - Formateo optimizado de respuestas
    """
    
    def __init__(self, model_path: str = None):
        """
        Inicializa el chatbot
        
        Args:
            model_path: Ruta al modelo (relativa o absoluta)
                       Si None, usa "models/best_rag_model" desde project root
        """
        # Convertir a ruta absoluta si es necesario
        if model_path is None:
            model_path = "models/best_rag_model"
        
        if not os.path.isabs(model_path):
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir.parent
            self.model_path = str(project_root / model_path)
        else:
            self.model_path = model_path
        
        self.config = None
        self.vectorstore = None
        self.retriever = None
        
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo RAG desde disco"""
        
        print(f"🔄 Cargando modelo RAG desde {self.model_path}...")
        
        # Cargar configuración
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config no encontrado: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        print(f"   ✅ Configuración cargada")
        
        # Recrear embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model'],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': self.config.get('normalize_embeddings', True)}
        )
        
        # Cargar vectorstore FAISS
        faiss_path = os.path.join(self.model_path, "faiss_index")
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index no encontrado: {faiss_path}")
        
        self.vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Aumentar K para luego reranquear
        k_value = self.config.get('k', 5)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k_value * 2}  # Recuperar más, luego filtrar
        )
        
        print(f"   ✅ Modelo cargado exitosamente\n")
    
    def _calculate_relevance_score(self, doc: Document, query: str) -> float:
        """
        Calcula un score de relevancia basado en coincidencias de términos
        
        Args:
            doc: Documento a evaluar
            query: Query del usuario
            
        Returns:
            Score de relevancia (mayor = más relevante)
        """
        doc_text = doc.page_content.lower()
        query_terms = query.lower().split()
        
        # Contar coincidencias de términos
        matches = sum(1 for term in query_terms if term in doc_text)
        
        # Normalizar por número de términos en la query
        score = matches / max(len(query_terms), 1)
        
        # Bonus si el documento es corto (más enfocado)
        if len(doc_text) < 500:
            score *= 1.2
        
        return score
    
    def _rerank_documents(self, docs: List[Document], query: str, top_k: int = 3) -> List[Document]:
        """
        Re-rankea documentos por relevancia real
        
        Mejora los resultados de la búsqueda vectorial inicial
        aplicando heurísticas de relevancia basadas en contenido
        
        Args:
            docs: Lista de documentos pre-recuperados
            query: Query original
            top_k: Número de documentos a devolver
            
        Returns:
            Lista de top_k documentos más relevantes
        """
        
        scored_docs = []
        for doc in docs:
            score = self._calculate_relevance_score(doc, query)
            scored_docs.append((doc, score))
        
        # Ordenar por score descendente
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver top K
        return [doc for doc, score in scored_docs[:top_k]]
    
    def _extract_key_info(self, text: str, max_chars: int = 400) -> str:
        """
        Extrae solo la información clave del texto
        
        Detecta si el texto contiene datos estructurados (tablas, listas)
        y ajusta la cantidad de texto a extraer en consecuencia
        
        Args:
            text: Texto completo del documento
            max_chars: Máximo de caracteres a extraer
            
        Returns:
            Texto extraído
        """
        
        # Si es corto, devolver completo
        if len(text) <= max_chars:
            return text
        
        # Buscar tablas o listas (tienen datos estructurados)
        if '|' in text or '\n-' in text or text.count('\n') > 5:
            # Es probable que tenga datos estructurados, tomar más
            return text[:max_chars * 2]
        
        # Texto normal: tomar primeros N caracteres
        return text[:max_chars] + "..."
    
    def _format_context(self, docs: List[Document]) -> str:
        """
        Formatea el contexto de manera clara y estructurada
        
        Preserva la estructura de tablas y datos tabulares
        mientras mantiene el formato legible
        
        Args:
            docs: Lista de documentos a formatear
            
        Returns:
            Contexto formateado como string
        """
        
        formatted_parts = []
        
        for i, doc in enumerate(docs, 1):
            article_num = doc.metadata.get('article_number', 'N/A')
            content = self._extract_key_info(doc.page_content, max_chars=300)
            
            # Detectar si es una tabla
            if '|' in content or (',' in content and content.count(',') > 5):
                # Formato de tabla - mantener estructura
                formatted_parts.append(f"[Article {article_num}]\n{content}")
            else:
                # Texto normal
                formatted_parts.append(f"[Article {article_num}] {content}")
        
        return "\n\n".join(formatted_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Genera una respuesta en lenguaje más natural a partir del contexto
        
        Usa templates y reglas para reformular el contexto en una respuesta
        más conversacional y directa, sin necesidad de LLMs externos
        
        Args:
            question: Pregunta del usuario
            context: Contexto recuperado de los documentos
            
        Returns:
            Respuesta en lenguaje natural
        """
        
        question_lower = question.lower()
        
        # Extraer información clave del contexto
        articles = self._extract_articles_from_context(context)
        
        # Construir respuesta según tipo de pregunta
        answer_parts = []
        
        # ====================================
        # PREGUNTAS DE CANTIDAD (how many)
        # ====================================
        if any(word in question_lower for word in ['how many', 'number of', 'quantity']):
            # Buscar números en el contexto
            import re
            numbers = re.findall(r'\b\d+\b', context)
            
            if numbers:
                answer_parts.append(f"According to the regulations, the answer is **{numbers[0]}**.")
            else:
                answer_parts.append("Based on the regulations:")
            
            # Añadir explicación simplificada
            answer_parts.append(self._simplify_context(context, max_sentences=3))
        
        # ====================================
        # PREGUNTAS DE PROCEDIMIENTO (what happens)
        # ====================================
        elif any(word in question_lower for word in ['what happens', 'what if', 'procedure']):
            answer_parts.append("Here's what happens according to the regulations:\n")
            
            # Si hay lista de puntos, mantenerla
            if '-' in context or '•' in context:
                answer_parts.append(self._format_list_naturally(context))
            else:
                answer_parts.append(self._simplify_context(context, max_sentences=4))
        
        # ====================================
        # PREGUNTAS DE PUNTOS/SCORING
        # ====================================
        elif any(word in question_lower for word in ['points', 'score', 'scoring']):
            answer_parts.append("The points allocation works as follows:\n")
            
            # Extraer y reformatear tabla/lista de puntos
            if any(keyword in context.lower() for keyword in ['points', 'awarded', 'classified']):
                answer_parts.append(self._format_points_table(context))
            else:
                answer_parts.append(context)
        
        # ====================================
        # PREGUNTAS DE TIEMPO/CUÁNDO (when)
        # ====================================
        elif any(word in question_lower for word in ['when', 'timing', 'time']):
            answer_parts.append("Regarding timing and conditions:\n")
            answer_parts.append(self._simplify_context(context, max_sentences=3))
        
        # ====================================
        # PREGUNTAS DE PENALIZACIONES
        # ====================================
        elif any(word in question_lower for word in ['penalty', 'penalties', 'sanction']):
            answer_parts.append("According to the penalty regulations:\n")
            answer_parts.append(self._simplify_context(context, max_sentences=4))
        
        # ====================================
        # PREGUNTAS GENERALES
        # ====================================
        else:
            answer_parts.append("Based on the FIA regulations:\n")
            answer_parts.append(self._simplify_context(context, max_sentences=4))
        
        # Añadir referencias a artículos si están disponibles
        if articles:
            answer_parts.append(f"\n*Reference: {', '.join(articles)}*")
        
        return "\n".join(answer_parts)
    
    def _extract_articles_from_context(self, context: str) -> List[str]:
        """Extrae números de artículos del contexto"""
        import re
        articles = re.findall(r'\[Article (\d+(?:\.\d+)?)\]', context)
        return [f"Article {art}" for art in articles]
    
    def _simplify_context(self, context: str, max_sentences: int = 3) -> str:
        """
        Simplifica el contexto eliminando metadatos y acortando
        
        Args:
            context: Contexto completo
            max_sentences: Máximo de oraciones a incluir
            
        Returns:
            Contexto simplificado
        """
        # Eliminar etiquetas de artículos del inicio
        import re
        text = re.sub(r'\[Article \d+(?:\.\d+)?\]\s*', '', context)
        
        # Dividir en oraciones
        sentences = []
        current = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Si es una línea de lista, tratarla como una oración
            if line.startswith('-') or line.startswith('•'):
                sentences.append(line)
            else:
                # Dividir por puntos
                parts = line.split('. ')
                sentences.extend([p.strip() + '.' if not p.endswith('.') else p.strip() for p in parts if p.strip()])
        
        # Tomar solo las primeras N oraciones
        result = '\n'.join(sentences[:max_sentences * 2])  # *2 porque listas cuentan como oraciones
        
        return result
    
    def _format_list_naturally(self, context: str) -> str:
        """Formatea listas de manera más natural"""
        lines = context.split('\n')
        formatted = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Eliminar etiquetas de artículos
            line = line.replace('[Article ', '').replace(']', '').strip()
            
            # Si comienza con número de artículo, saltarlo
            if line and line[0].isdigit() and line[1:3] in ['. ', ' ']:
                continue
            
            # Reformular bullets
            if line.startswith('-'):
                # Cambiar "- " por "• "
                line = '• ' + line[1:].strip()
            
            formatted.append(line)
        
        return '\n'.join(formatted)
    
    def _format_points_table(self, context: str) -> str:
        """
        Formatea tablas de puntos de manera más legible
        
        Args:
            context: Contexto con información de puntos
            
        Returns:
            Tabla formateada
        """
        lines = context.split('\n')
        formatted = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Eliminar etiquetas de artículos
            line = line.replace('[Article ', '').replace(']', '')
            
            # Si es un encabezado, resaltarlo
            if any(word in line.lower() for word in ['points', 'allocation', 'system']):
                continue  # Saltar encabezados redundantes
            
            # Si es una condición de puntos, formatearla
            if any(symbol in line for symbol in [':', '-', '•']):
                # Ya está formateada, mantenerla
                formatted.append(line)
            else:
                formatted.append(line)
        
        return '\n'.join(formatted)
    
    def query(
        self, 
        question: str, 
        return_sources: bool = True,
        max_results: int = 3,
        max_total_chars: int = 800,
        generate_answer: bool = True
    ) -> Dict:
        """
        Realiza una consulta optimizada al chatbot
        
        Proceso:
        1. Recupera documentos usando búsqueda vectorial
        2. Re-rankea por relevancia real
        3. Formatea y limita el contexto
        4. Genera respuesta estructurada (opcional)
        5. Retorna resultado estructurado
        
        Args:
            question: Pregunta del usuario
            return_sources: Si True, devuelve los documentos fuente completos
            max_results: Máximo número de documentos a devolver
            max_total_chars: Máximo de caracteres en el contexto total
            generate_answer: Si True, genera respuesta estructurada
            
        Returns:
            Dict con:
                - question: La pregunta original
                - answer: Respuesta generada (si generate_answer=True)
                - context: Contexto formateado (raw)
                - num_sources: Número de fuentes usadas
                - sources: (opcional) Lista de documentos fuente
        """
        
        # Recuperar documentos (K alto para tener más candidatos)
        docs = self.retriever.invoke(question)
        
        # Re-rankear por relevancia
        docs = self._rerank_documents(docs, question, top_k=max_results)
        
        # Formatear contexto
        context = self._format_context(docs)
        
        # Limitar tamaño total
        if len(context) > max_total_chars:
            context = context[:max_total_chars] + "\n..."
        
        # Preparar respuesta
        response = {
            'question': question,
            'context': context,
            'num_sources': len(docs),
        }
        
        # Generar respuesta estructurada si se solicita
        if generate_answer:
            response['answer'] = self._generate_answer(question, context)
        
        if return_sources:
            sources = []
            for doc in docs:
                sources.append({
                    'article_number': doc.metadata.get('article_number', 'N/A'),
                    'content': doc.page_content,
                    'source_file': doc.metadata.get('source', 'N/A'),
                    'category': doc.metadata.get('category', 'N/A')
                })
            response['sources'] = sources
        
        return response
    
    def get_info(self) -> Dict:
        """
        Obtiene información del modelo cargado
        
        Returns:
            Dict con métricas y configuración del modelo
        """
        
        return {
            'model_name': self.config.get('model_name'),
            'embedding_model': self.config.get('embedding_model'),
            'chunk_size': self.config.get('chunk_size'),
            'k': self.config.get('k'),
            'search_type': self.config.get('search_type'),
            'performance_metrics': self.config.get('performance_metrics', {}),
            'total_chunks': self.config.get('total_chunks'),
            'created_at': self.config.get('created_at'),
            'chunking_strategy': self.config.get('chunking_strategy', 'N/A')
        }
    
    def interactive_mode(self):
        """
        Modo interactivo por consola
        
        Permite al usuario hacer preguntas de forma interactiva
        y ver los resultados inmediatamente
        """
        
        print("\n" + "="*80)
        print("🏎️  F1 REGULATIONS 2025 - CHATBOT")
        print("="*80)
        print(f"Model: {self.config.get('model_name', 'N/A')}")
        
        perf = self.config.get('performance_metrics', {})
        if 'f1_score' in perf:
            print(f"F1 Score: {perf['f1_score']:.3f}")
        
        print("\nCommands:")
        print("  - Type your question in English")
        print("  - Type 'info' to see model information")
        print("  - Type 'raw' to toggle raw context mode")
        print("  - Type 'exit' or 'quit' to exit")
        print("="*80 + "\n")
        
        show_raw = False
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if question.lower() == 'info':
                    info = self.get_info()
                    print(f"\n📊 Model Information:")
                    for key, value in info.items():
                        print(f"   {key}: {value}")
                    print()
                    continue
                
                if question.lower() == 'raw':
                    show_raw = not show_raw
                    mode = "RAW CONTEXT" if show_raw else "FORMATTED ANSWER"
                    print(f"\n🔄 Mode switched to: {mode}\n")
                    continue
                
                if not question:
                    continue
                
                print(f"\n🔍 Searching regulations...")
                result = self.query(
                    question, 
                    max_results=3, 
                    max_total_chars=800,
                    generate_answer=not show_raw
                )
                
                print(f"\n📄 {'Answer' if not show_raw else 'Raw Context'}:\n")
                
                if show_raw:
                    print(result['context'])
                else:
                    print(result.get('answer', result['context']))
                
                print(f"\n📊 Sources: {result['num_sources']}")
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

# ====================================
# EJEMPLO DE USO
# ====================================

def main():
    """Test del chatbot con consulta de ejemplo"""
    
    # Inicializar chatbot (ruta relativa se convierte automáticamente a absoluta)
    chatbot = F1RAGChatbot(model_path="models/best_rag_model")
    
    print("\n" + "="*80)
    print("TEST: Consulta de ejemplo")
    print("="*80 + "\n")
    
    # Realizar consulta de prueba
    question = "If race is not completed to 70% indicate points and positions"
    
    print(f"Question: {question}\n")
    print("🔍 Searching regulations...\n")
    
    # Obtener respuesta con formato estructurado
    result = chatbot.query(
        question,
        max_results=3,
        max_total_chars=800,
        generate_answer=True  # Generar respuesta formateada
    )
    
    print("="*80)
    print("📄 FORMATTED ANSWER:")
    print("="*80)
    print(result.get('answer', result['context']))
    print("\n" + "="*80)
    print("📚 RAW CONTEXT (for debugging):")
    print("="*80)
    print(result['context'])
    print("\n" + "="*80)
    print(f"📊 Sources used: {result['num_sources']}")
    print("="*80 + "\n")
    
    # Mostrar ejemplo de modo interactivo
    print("💡 TIP: Run chatbot.interactive_mode() for an interactive session")
    print("        Type 'raw' in interactive mode to toggle between formatted/raw output\n")
    
    # Modo interactivo (descomentar para usar)
    # chatbot.interactive_mode()

if __name__ == "__main__":
    main()