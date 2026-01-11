# ====================================
# evaluate_rag_strategies_with_models.py
# Script completo para evaluar estrategias de RAG con dataset desde JSON
# Corregido: rutas absolutas basadas en ubicación del script
# ====================================

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import sys
from collections import defaultdict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ====================================
# CONFIGURACIÓN
# ====================================

@dataclass
class RAGConfig:
    """
    Configuración de parámetros del sistema RAG
    
    Attributes:
        chunk_size: Tamaño de cada chunk de texto
        chunk_overlap: Solapamiento entre chunks consecutivos
        separators: Lista de separadores para dividir texto
        embedding_model: Modelo de embeddings a utilizar
        normalize_embeddings: Si normalizar los embeddings
        k: Número de documentos a recuperar
        search_type: Tipo de búsqueda ('similarity' o 'mmr')
        fetch_k: Documentos a recuperar antes de MMR
        lambda_mult: Factor de diversidad para MMR (0=diversidad, 1=relevancia)
    """
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    normalize_embeddings: bool = True
    k: int = 3
    search_type: str = "similarity"
    fetch_k: int = 20
    lambda_mult: float = 0.5
    
    def __post_init__(self):
        """Inicializa separadores por defecto si no se proporcionan"""
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]

# ====================================
# CATÁLOGO DE MODELOS
# ====================================

class EmbeddingModels:
    """
    Catálogo de modelos de embeddings disponibles
    
    Cada modelo incluye:
    - name: Nombre completo del modelo en HuggingFace
    - description: Descripción breve
    - languages: Idiomas soportados
    - dim: Dimensiones del vector de embedding
    - speed: Velocidad relativa (very_fast, fast, medium, slow)
    - size: Tamaño del modelo en disco
    """
    
    MODELS = {
        'multilingual-mini': {
            'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'description': 'Multilingüe, rápido, ligero (118MB)',
            'languages': ['es', 'en'],
            'dim': 384,
            'speed': 'fast',
            'size': 'small'
        },
        'multilingual-mpnet': {
            'name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'description': 'Multilingüe, mejor calidad (970MB)',
            'languages': ['es', 'en'],
            'dim': 768,
            'speed': 'medium',
            'size': 'large'
        },
        'all-mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'description': 'Inglés, alta calidad (420MB)',
            'languages': ['en'],
            'dim': 768,
            'speed': 'medium',
            'size': 'medium'
        },
        'all-minilm': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'description': 'Inglés, muy rápido (80MB)',
            'languages': ['en'],
            'dim': 384,
            'speed': 'very_fast',
            'size': 'tiny'
        },
        'bge-small': {
            'name': 'BAAI/bge-small-en-v1.5',
            'description': 'Inglés, excelente rendimiento (133MB)',
            'languages': ['en'],
            'dim': 384,
            'speed': 'fast',
            'size': 'small'
        },
        'bge-base': {
            'name': 'BAAI/bge-base-en-v1.5',
            'description': 'Inglés, top performance (420MB)',
            'languages': ['en'],
            'dim': 768,
            'speed': 'medium',
            'size': 'medium'
        },
    }
    
    @classmethod
    def get_model_info(cls, model_key: str) -> Dict:
        """Obtiene información de un modelo por su clave"""
        return cls.MODELS.get(model_key, {})
    
    @classmethod
    def print_catalog(cls):
        """Imprime el catálogo completo de modelos"""
        print("\n" + "="*100)
        print("📚 CATÁLOGO DE MODELOS DE EMBEDDINGS")
        print("="*100 + "\n")
        
        for key, info in cls.MODELS.items():
            print(f"🔹 {key}")
            print(f"   Modelo: {info['name']}")
            print(f"   Descripción: {info['description']}")
            print(f"   Idiomas: {', '.join(info['languages'])} | "
                  f"Dimensiones: {info['dim']} | "
                  f"Velocidad: {info['speed']} | "
                  f"Tamaño: {info['size']}")
            print()

# ====================================
# MÉTRICAS
# ====================================

@dataclass
class RAGMetrics:
    """
    Métricas completas para evaluación de sistemas RAG
    
    Incluye:
    - Métricas de precisión/recall
    - Métricas de ranking (MRR, MAP, NDCG)
    - Métricas de cobertura y redundancia
    - Estadísticas de contexto
    - Tiempos de ejecución
    - Distribuciones y diversidad
    """
    
    # Métricas básicas
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_score: float = 0.0
    
    # Métricas de ranking
    mrr: float = 0.0  # Mean Reciprocal Rank
    map_score: float = 0.0  # Mean Average Precision
    ndcg_at_k: float = 0.0  # Normalized Discounted Cumulative Gain
    
    # Precisión a diferentes K
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    
    # Métricas de cobertura
    coverage: float = 0.0  # % de artículos relevantes recuperados
    redundancy: float = 0.0  # % de documentos duplicados
    
    # Estadísticas de contexto
    avg_context_length: float = 0.0
    min_context_length: float = 0.0
    max_context_length: float = 0.0
    std_context_length: float = 0.0
    context_diversity: float = 0.0  # Diversidad de categorías
    
    # Distribuciones
    category_distribution: Dict[str, int] = None
    article_distribution: Dict[int, int] = None
    
    # Tiempos
    retrieval_time: float = 0.0
    embedding_time: float = 0.0
    
    # Información del índice
    num_chunks_retrieved: int = 0
    total_chunks_indexed: int = 0
    
    # Scores de similitud
    avg_similarity_score: float = 0.0
    min_similarity_score: float = 0.0
    max_similarity_score: float = 0.0
    
    def __post_init__(self):
        """Inicializa diccionarios vacíos si no se proporcionan"""
        if self.category_distribution is None:
            self.category_distribution = {}
        if self.article_distribution is None:
            self.article_distribution = {}
    
    def to_dict(self):
        """Convierte las métricas a diccionario"""
        return asdict(self)

# ====================================
# DATASET DESDE JSON
# ====================================

class F1TestDataset:
    """
    Dataset de prueba cargado desde archivo JSON
    
    Estructura esperada del JSON:
    {
        "dataset_info": {
            "name": "...",
            "version": "...",
            "language": "..."
        },
        "test_cases": [
            {
                "query_id": "...",
                "query": "...",
                "relevant_articles": [1, 2, 3],
                "category": "...",
                "difficulty": "easy|medium|hard"
            }
        ]
    }
    """
    
    def __init__(self, json_path: str = None):
        """
        Inicializa el dataset
        
        Args:
            json_path: Ruta al archivo JSON (si es relativa, se convierte a absoluta)
        """
        # Convertir ruta relativa a absoluta basada en ubicación del script
        if json_path is None:
            json_path = "data/test_dataset.json"
        
        if not os.path.isabs(json_path):
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir.parent
            self.json_path = str(project_root / json_path)
        else:
            self.json_path = json_path
            
        self.dataset_info = {}
        self.test_cases = []
        self._load_from_json()
    
    def _load_from_json(self):
        """Carga el dataset desde JSON"""
        
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(
                f"❌ Dataset JSON no encontrado: {self.json_path}\n"
                f"   Por favor, crea el archivo con las queries de test."
            )
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.dataset_info = data.get('dataset_info', {})
            self.test_cases = data.get('test_cases', [])
            
            print(f"✅ Dataset cargado: {self.dataset_info.get('name', 'Unknown')}")
            print(f"   Versión: {self.dataset_info.get('version', 'N/A')}")
            print(f"   Idioma: {self.dataset_info.get('language', 'N/A')}")
            print(f"   Total queries: {len(self.test_cases)}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Error parsing JSON: {e}")
        except Exception as e:
            raise Exception(f"❌ Error cargando dataset: {e}")
    
    def get_test_cases(self):
        """Obtiene todos los casos de prueba"""
        return self.test_cases
    
    def get_by_difficulty(self, difficulty: str):
        """Filtra casos por dificultad"""
        return [tc for tc in self.test_cases if tc.get('difficulty') == difficulty]
    
    def get_by_category(self, category: str):
        """Filtra casos por categoría"""
        return [tc for tc in self.test_cases if tc.get('category') == category]
    
    def get_by_id(self, query_id: str):
        """Obtiene un caso específico por ID"""
        for tc in self.test_cases:
            if tc.get('query_id') == query_id:
                return tc
        return None
    
    def get_statistics(self):
        """Calcula estadísticas del dataset"""
        total = len(self.test_cases)
        by_difficulty = defaultdict(int)
        by_category = defaultdict(int)
        
        for tc in self.test_cases:
            by_difficulty[tc.get('difficulty', 'unknown')] += 1
            by_category[tc.get('category', 'unknown')] += 1
        
        return {
            'total_queries': total,
            'by_difficulty': dict(by_difficulty),
            'by_category': dict(by_category),
            'avg_relevant_articles': float(np.mean([len(tc.get('relevant_articles', [])) for tc in self.test_cases])),
            'queries_with_multiple_articles': int(sum(1 for tc in self.test_cases if len(tc.get('relevant_articles', [])) > 1)),
            'dataset_info': self.dataset_info
        }
    
    def validate_dataset(self) -> List[str]:
        """
        Valida la integridad del dataset
        
        Returns:
            Lista de errores encontrados (vacía si todo OK)
        """
        errors = []
        
        if not self.test_cases:
            errors.append("Dataset vacío")
            return errors
        
        # Validar cada caso de prueba
        for i, tc in enumerate(self.test_cases, 1):
            # Campos requeridos
            required = ['query_id', 'query', 'relevant_articles', 'category', 'difficulty']
            for field in required:
                if field not in tc:
                    errors.append(f"Query {i}: Falta campo '{field}'")
            
            # IDs únicos
            query_ids = [t.get('query_id') for t in self.test_cases]
            if query_ids.count(tc.get('query_id')) > 1:
                errors.append(f"Query {i}: query_id duplicado '{tc.get('query_id')}'")
            
            # Al menos un artículo relevante
            if not tc.get('relevant_articles'):
                errors.append(f"Query {i}: relevant_articles vacío")
            
            # Dificultad válida
            valid_difficulties = ['easy', 'medium', 'hard']
            if tc.get('difficulty') not in valid_difficulties:
                errors.append(f"Query {i}: difficulty debe ser {valid_difficulties}")
        
        if not errors:
            print("✅ Dataset validado correctamente")
        else:
            print(f"⚠️  Se encontraron {len(errors)} errores:")
            for error in errors[:5]:  # Mostrar solo primeros 5
                print(f"   - {error}")
        
        return errors

# ====================================
# EVALUADOR
# ====================================

class AdvancedRetrievalEvaluator:
    """
    Evaluador con métricas extendidas para sistemas RAG
    
    Implementa:
    - Precision/Recall/F1
    - MRR (Mean Reciprocal Rank)
    - MAP (Mean Average Precision)
    - NDCG (Normalized Discounted Cumulative Gain)
    - Coverage y Redundancy
    - Estadísticas de contexto
    """
    
    def calculate_precision_recall(self, retrieved_docs: List[Document], relevant_articles: List[int]) -> Tuple[float, float, float]:
        """
        Calcula precision, recall y F1 score
        
        Args:
            retrieved_docs: Documentos recuperados
            relevant_articles: Lista de artículos relevantes
            
        Returns:
            Tupla (precision, recall, f1_score)
        """
        if not retrieved_docs:
            return 0.0, 0.0, 0.0
        
        # Extraer números de artículo de los documentos recuperados
        retrieved_articles = []
        for doc in retrieved_docs:
            article_num = self._extract_article_number(doc)
            if article_num != -1:
                retrieved_articles.append(article_num)
        
        # Calcular intersección
        relevant_retrieved = set(retrieved_articles) & set(relevant_articles)
        
        # Métricas
        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
        recall = len(relevant_retrieved) / len(relevant_articles) if relevant_articles else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_precision_at_k(self, retrieved_docs: List[Document], relevant_articles: List[int], k_values: List[int] = [1, 3, 5]) -> Dict[int, float]:
        """
        Calcula precision@k para diferentes valores de k
        
        Args:
            retrieved_docs: Documentos recuperados
            relevant_articles: Artículos relevantes
            k_values: Valores de k a evaluar
            
        Returns:
            Diccionario {k: precision@k}
        """
        precisions = {}
        
        for k in k_values:
            docs_at_k = retrieved_docs[:k]
            retrieved_at_k = []
            
            for doc in docs_at_k:
                article_num = self._extract_article_number(doc)
                if article_num != -1:
                    retrieved_at_k.append(article_num)
            
            relevant_at_k = set(retrieved_at_k) & set(relevant_articles)
            precisions[k] = len(relevant_at_k) / k if k > 0 else 0.0
        
        return precisions
    
    def calculate_mrr(self, retrieved_docs: List[Document], relevant_articles: List[int]) -> float:
        """
        Mean Reciprocal Rank: 1 / posición_primer_relevante
        
        Mide qué tan arriba aparece el primer documento relevante
        """
        for i, doc in enumerate(retrieved_docs, 1):
            article_num = self._extract_article_number(doc)
            if article_num in relevant_articles:
                return 1.0 / i
        return 0.0
    
    def calculate_map(self, retrieved_docs: List[Document], relevant_articles: List[int]) -> float:
        """
        Mean Average Precision
        
        Promedio de precisiones en cada posición donde hay un documento relevante
        """
        precisions = []
        num_relevant = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            article_num = self._extract_article_number(doc)
            if article_num in relevant_articles:
                num_relevant += 1
                precision_at_i = num_relevant / i
                precisions.append(precision_at_i)
        
        return float(np.mean(precisions)) if precisions else 0.0
    
    def calculate_ndcg(self, retrieved_docs: List[Document], highly_relevant: List[int], partially_relevant: List[int], k: int = None) -> float:
        """
        Normalized Discounted Cumulative Gain
        
        Considera diferentes niveles de relevancia y penaliza documentos relevantes
        que aparecen más abajo en el ranking
        
        Args:
            retrieved_docs: Documentos recuperados
            highly_relevant: Artículos muy relevantes (score 2)
            partially_relevant: Artículos parcialmente relevantes (score 1)
            k: Considerar solo top-k documentos
        """
        if k is None:
            k = len(retrieved_docs)
        
        # Asignar scores de relevancia
        relevance_scores = []
        for doc in retrieved_docs[:k]:
            article_num = self._extract_article_number(doc)
            if article_num in highly_relevant:
                relevance_scores.append(2)
            elif article_num in partially_relevant:
                relevance_scores.append(1)
            else:
                relevance_scores.append(0)
        
        # DCG (Discounted Cumulative Gain)
        dcg = relevance_scores[0] if len(relevance_scores) > 0 else 0
        for i, score in enumerate(relevance_scores[1:], 2):
            dcg += score / np.log2(i)
        
        # IDCG (Ideal DCG) - mejor ordenamiento posible
        ideal_scores = sorted([2] * len(highly_relevant) + [1] * len(partially_relevant), reverse=True)[:k]
        idcg = ideal_scores[0] if len(ideal_scores) > 0 else 0
        for i, score in enumerate(ideal_scores[1:], 2):
            idcg += score / np.log2(i)
        
        # NDCG normalizado
        return float(dcg / idcg) if idcg > 0 else 0.0
    
    def calculate_coverage(self, retrieved_docs: List[Document], relevant_articles: List[int]) -> float:
        """
        Coverage: % de artículos relevantes que fueron recuperados
        
        Mide qué tan completo es el resultado
        """
        retrieved_articles = set()
        for doc in retrieved_docs:
            article_num = self._extract_article_number(doc)
            if article_num != -1:
                retrieved_articles.add(article_num)
        
        relevant_set = set(relevant_articles)
        found = retrieved_articles & relevant_set
        
        return len(found) / len(relevant_set) if relevant_set else 0.0
    
    def calculate_redundancy(self, retrieved_docs: List[Document]) -> float:
        """
        Redundancy: % de documentos que son duplicados (mismo artículo)
        
        Valores altos indican que se recuperan múltiples chunks del mismo artículo
        """
        article_counts = defaultdict(int)
        for doc in retrieved_docs:
            article_num = self._extract_article_number(doc)
            if article_num != -1:
                article_counts[article_num] += 1
        
        total_docs = len(retrieved_docs)
        unique_docs = len(article_counts)
        
        return (total_docs - unique_docs) / total_docs if total_docs > 0 else 0.0
    
    def calculate_diversity(self, retrieved_docs: List[Document]) -> float:
        """
        Diversity: Diversidad de categorías en los documentos recuperados
        
        Valores altos indican mayor variedad de temas
        """
        if not retrieved_docs:
            return 0.0
        
        categories = [doc.metadata.get('category', 'unknown') for doc in retrieved_docs]
        unique_categories = len(set(categories))
        
        return unique_categories / len(categories)
    
    def calculate_context_stats(self, retrieved_docs: List[Document]) -> Dict[str, float]:
        """
        Estadísticas sobre la longitud del contexto recuperado
        
        Returns:
            Dict con avg, min, max, std de longitud de documentos
        """
        if not retrieved_docs:
            return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        lengths = [len(doc.page_content) for doc in retrieved_docs]
        
        return {
            'avg': float(np.mean(lengths)),
            'min': float(np.min(lengths)),
            'max': float(np.max(lengths)),
            'std': float(np.std(lengths))
        }
    
    def _extract_article_number(self, doc: Document) -> int:
        """
        Extrae el número de artículo de un documento
        
        Intenta primero desde metadata, luego desde el nombre del archivo
        
        Returns:
            Número de artículo o -1 si no se encuentra
        """
        # Intentar desde metadata
        article_num = doc.metadata.get('article_number')
        if article_num is not None:
            return article_num
        
        # Intentar desde source filename (formato: SR2025_A##_...)
        source = doc.metadata.get('source', '')
        if 'A' in source:
            try:
                return int(source.split('_A')[1].split('_')[0])
            except:
                pass
        
        return -1
    
    def evaluate_query(self, retrieved_docs: List[Document], test_case: Dict, retrieval_time: float, total_chunks: int, similarity_scores: List[float] = None) -> RAGMetrics:
        """
        Evalúa una query completa calculando todas las métricas
        
        Args:
            retrieved_docs: Documentos recuperados por el sistema
            test_case: Caso de prueba con ground truth
            retrieval_time: Tiempo de recuperación en segundos
            total_chunks: Total de chunks en el índice
            similarity_scores: Scores de similitud de los documentos
            
        Returns:
            RAGMetrics con todas las métricas calculadas
        """
        # Extraer ground truth
        relevant_articles = test_case['relevant_articles']
        highly_relevant = test_case.get('highly_relevant', relevant_articles)
        partially_relevant = test_case.get('partially_relevant', [])
        
        # Calcular métricas básicas
        precision, recall, f1 = self.calculate_precision_recall(retrieved_docs, relevant_articles)
        mrr = self.calculate_mrr(retrieved_docs, relevant_articles)
        map_score = self.calculate_map(retrieved_docs, relevant_articles)
        ndcg = self.calculate_ndcg(retrieved_docs, highly_relevant, partially_relevant)
        
        # Precision@k
        p_at_k = self.calculate_precision_at_k(retrieved_docs, relevant_articles, [1, 3, 5])
        
        # Métricas de cobertura
        coverage = self.calculate_coverage(retrieved_docs, relevant_articles)
        redundancy = self.calculate_redundancy(retrieved_docs)
        diversity = self.calculate_diversity(retrieved_docs)
        
        # Estadísticas de contexto
        context_stats = self.calculate_context_stats(retrieved_docs)
        
        # Estadísticas de similitud
        avg_sim = float(np.mean(similarity_scores)) if similarity_scores else 0.0
        min_sim = float(np.min(similarity_scores)) if similarity_scores else 0.0
        max_sim = float(np.max(similarity_scores)) if similarity_scores else 0.0
        
        # Crear objeto de métricas
        metrics = RAGMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            f1_score=f1,
            mrr=mrr,
            map_score=map_score,
            ndcg_at_k=ndcg,
            precision_at_1=p_at_k.get(1, 0.0),
            precision_at_3=p_at_k.get(3, 0.0),
            precision_at_5=p_at_k.get(5, 0.0),
            coverage=coverage,
            redundancy=redundancy,
            avg_context_length=context_stats['avg'],
            min_context_length=context_stats['min'],
            max_context_length=context_stats['max'],
            std_context_length=context_stats['std'],
            context_diversity=diversity,
            retrieval_time=retrieval_time,
            num_chunks_retrieved=len(retrieved_docs),
            total_chunks_indexed=total_chunks,
            avg_similarity_score=avg_sim,
            min_similarity_score=min_sim,
            max_similarity_score=max_sim
        )
        
        return metrics

# ====================================
# SISTEMA RAG
# ====================================

class ConfigurableRAG:
    """
    Sistema RAG con parámetros configurables
    
    Permite experimentar con diferentes:
    - Modelos de embeddings
    - Tamaños de chunks
    - Estrategias de búsqueda
    - Parámetros de retrieval
    """
    
    def __init__(self, config: RAGConfig):
        """
        Inicializa el sistema RAG
        
        Args:
            config: Configuración del sistema RAG
        """
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.total_chunks = 0
        self.embedding_creation_time = 0.0
        
    def load_documents(self, folder_path: str) -> List[Document]:
        """
        Carga todos los documentos de texto de una carpeta
        
        Args:
            folder_path: Ruta a la carpeta con documentos .txt y .md
            
        Returns:
            Lista de Documents de LangChain
        """
        documents = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt") or filename.endswith(".md"):
                try:
                    file_path = os.path.join(folder_path, filename)
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                    # Extraer número de artículo del nombre del archivo
                    for doc in docs:
                        if 'A' in filename:
                            try:
                                article_num = int(filename.split('_A')[1].split('_')[0])
                                doc.metadata['article_number'] = article_num
                            except:
                                pass
                    
                    documents.extend(docs)
                except Exception as e:
                    pass  # Ignorar archivos que no se puedan leer
        
        return documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Divide documentos en chunks según configuración
        
        Args:
            documents: Lista de documentos completos
            
        Returns:
            Lista de chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        self.total_chunks = len(chunks)
        
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]):
        """
        Crea el vector store con embeddings
        
        Args:
            chunks: Lista de chunks de documentos
        """
        start_time = time.time()
        
        # Crear embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': self.config.normalize_embeddings}
        )
        
        # Crear FAISS index
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        self.embedding_creation_time = time.time() - start_time
        
        # Configurar retriever según tipo de búsqueda
        if self.config.search_type == "similarity":
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.k}
            )
        elif self.config.search_type == "mmr":
            # MMR (Maximal Marginal Relevance) para diversidad
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.config.k,
                    "fetch_k": self.config.fetch_k,
                    "lambda_mult": self.config.lambda_mult
                }
            )
    
    def retrieve_documents(self, query: str) -> Tuple[List[Document], float, List[float]]:
        """
        Recupera documentos relevantes para una query
        
        Args:
            query: Pregunta del usuario
            
        Returns:
            Tupla (documentos, tiempo_retrieval, similarity_scores)
        """
        start_time = time.time()
        
        # Búsqueda con scores de similitud
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.config.k)
        
        retrieval_time = time.time() - start_time
        
        # Separar documentos y scores
        docs = [doc for doc, score in docs_and_scores]
        scores = [float(score) for doc, score in docs_and_scores]
        
        return docs, retrieval_time, scores

# ====================================
# COMPARADOR MULTI-MODELO
# ====================================

class MultiModelRAGComparator:
    """
    Comparador que prueba diferentes modelos y configuraciones
    
    Permite evaluar sistemáticamente:
    - Diferentes modelos de embeddings
    - Diferentes tamaños de chunks
    - Diferentes estrategias de búsqueda
    """
    
    def __init__(self, folder_path: str, dataset_path: str = "data/test_dataset.json"):
        """
        Inicializa el comparador
        
        Args:
            folder_path: Ruta a documentos procesados (relativa o absoluta)
            dataset_path: Ruta al dataset JSON (relativa o absoluta)
        """
        # Convertir rutas relativas a absolutas
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        
        if not os.path.isabs(folder_path):
            self.folder_path = str(project_root / folder_path.lstrip('../'))
        else:
            self.folder_path = folder_path
            
        print(f"🔧 Cargando documentos desde: {self.folder_path}")
        
        self.test_dataset = F1TestDataset(dataset_path)
        self.evaluator = AdvancedRetrievalEvaluator()
        self.results = []
        
    def _convert_to_python_types(self, value):
        """
        Convierte numpy types a Python types para serialización JSON
        
        Args:
            value: Valor a convertir
            
        Returns:
            Valor convertido a tipo Python nativo
        """
        if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._convert_to_python_types(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_to_python_types(item) for item in value]
        else:
            return value
    
    def test_configuration(self, config_name: str, config: RAGConfig) -> Dict:
        """
        Prueba una configuración específica de RAG
        
        Args:
            config_name: Nombre descriptivo de la configuración
            config: Configuración RAG a probar
            
        Returns:
            Diccionario con resultados de la evaluación
        """
        
        print(f"\n{'='*80}")
        print(f"🧪 Probando: {config_name}")
        print(f"{'='*80}")
        print(f"   Modelo: {config.embedding_model.split('/')[-1]}")
        print(f"   Chunk size: {config.chunk_size} | Overlap: {config.chunk_overlap}")
        print(f"   K: {config.k} | Search: {config.search_type}")
        
        try:
            # Crear sistema RAG
            rag = ConfigurableRAG(config)
            
            # Cargar y procesar documentos
            print(f"   📂 Cargando documentos...")
            documents = rag.load_documents(self.folder_path)
            
            print(f"   ✂️  Creando chunks...")
            chunks = rag.create_chunks(documents)
            
            print(f"   🔄 Creando embeddings...")
            rag.create_vectorstore(chunks)
            
            print(f"   ✅ Embeddings creados en {rag.embedding_creation_time:.2f}s")
            
            # Evaluar todas las queries del dataset
            metrics_list = []
            test_cases = self.test_dataset.get_test_cases()
            
            print(f"   📋 Evaluando {len(test_cases)} queries...")
            
            for i, test_case in enumerate(test_cases, 1):
                query = test_case['query']
                
                # Recuperar documentos
                docs, retrieval_time, similarity_scores = rag.retrieve_documents(query)
                
                # Evaluar
                metrics = self.evaluator.evaluate_query(
                    docs,
                    test_case,
                    retrieval_time,
                    rag.total_chunks,
                    similarity_scores
                )
                
                metrics.embedding_time = rag.embedding_creation_time
                
                metrics_list.append({
                    'query_id': test_case['query_id'],
                    'difficulty': test_case['difficulty'],
                    'category': test_case['category'],
                    'metrics': metrics.to_dict()
                })
            
            # Calcular métricas promedio
            avg_metrics = self._calculate_average_metrics(metrics_list)
            
            # Obtener info del modelo
            model_info = None
            for key, info in EmbeddingModels.MODELS.items():
                if info['name'] == config.embedding_model:
                    model_info = info
                    break
            
            # Crear resultado
            result = {
                'config_name': config_name,
                'config': {
                    'chunk_size': config.chunk_size,
                    'chunk_overlap': config.chunk_overlap,
                    'k': config.k,
                    'search_type': config.search_type,
                    'embedding_model': config.embedding_model,
                    'model_info': model_info
                },
                'avg_metrics': avg_metrics,
                'individual_results': metrics_list
            }
            
            self.results.append(result)
            
            print(f"   📊 F1: {avg_metrics.get('f1_score', 0):.3f} | "
                  f"NDCG: {avg_metrics.get('ndcg_at_k', 0):.3f} | "
                  f"P@1: {avg_metrics.get('precision_at_1', 0):.3f}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """
        Calcula métricas promedio de múltiples evaluaciones
        
        Args:
            metrics_list: Lista de resultados individuales
            
        Returns:
            Diccionario con métricas promedio
        """
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        sample_dict = metrics_list[0]['metrics']
        
        # Promediar valores numéricos
        for key in sample_dict.keys():
            if key in ['category_distribution', 'article_distribution']:
                continue  # Ignorar distribuciones
                
            values = [m['metrics'][key] for m in metrics_list]
            avg_value = np.mean(values)
            avg_metrics[key] = self._convert_to_python_types(avg_value)
        
        return avg_metrics
    
    def compare_models_and_strategies(self, embedding_models: List[str], chunking_strategies: List[Dict]):
        """
        Compara combinaciones de modelos y estrategias de chunking
        
        Args:
            embedding_models: Lista de claves de modelos a probar
            chunking_strategies: Lista de dicts con configuraciones de chunking
        """
        
        print(f"\n{'='*100}")
        print(f"🏁 COMPARACIÓN MULTI-MODELO")
        print(f"{'='*100}")
        print(f"   Modelos a probar: {len(embedding_models)}")
        print(f"   Estrategias de chunking: {len(chunking_strategies)}")
        print(f"   Total configuraciones: {len(embedding_models) * len(chunking_strategies)}")
        print(f"{'='*100}\n")
        
        total_configs = len(embedding_models) * len(chunking_strategies)
        current = 0
        
        # Probar cada combinación
        for model_key in embedding_models:
            model_info = EmbeddingModels.get_model_info(model_key)
            
            if not model_info:
                print(f"⚠️  Modelo '{model_key}' no encontrado")
                continue
            
            for strategy in chunking_strategies:
                current += 1
                
                # Crear configuración
                config = RAGConfig()
                config.embedding_model = model_info['name']
                config.chunk_size = strategy.get('chunk_size', 1000)
                config.chunk_overlap = strategy.get('chunk_overlap', 200)
                config.k = strategy.get('k', 3)
                config.search_type = strategy.get('search_type', 'similarity')
                
                config_name = f"{model_key} | Chunks {config.chunk_size}"
                
                print(f"\n[{current}/{total_configs}] ", end="")
                self.test_configuration(config_name, config)
        
        # Generar reportes
        self.print_comparison_by_model()
        self.print_comparison_by_chunking()
        self.print_best_combinations()
        self.save_results()
        
        return self.results
    
    def print_comparison_by_model(self):
        """Compara rendimiento por modelo de embedding"""
        
        print(f"\n{'='*100}")
        print(f"📊 COMPARACIÓN POR MODELO DE EMBEDDING")
        print(f"{'='*100}\n")
        
        # Agrupar por modelo
        by_model = defaultdict(list)
        for result in self.results:
            model_name = result['config']['embedding_model'].split('/')[-1]
            by_model[model_name].append(result)
        
        # Calcular estadísticas por modelo
        model_stats = []
        for model_name, results in by_model.items():
            avg_f1 = float(np.mean([r['avg_metrics'].get('f1_score', 0) for r in results]))
            avg_ndcg = float(np.mean([r['avg_metrics'].get('ndcg_at_k', 0) for r in results]))
            avg_p1 = float(np.mean([r['avg_metrics'].get('precision_at_1', 0) for r in results]))
            avg_time = float(np.mean([r['avg_metrics'].get('embedding_time', 0) for r in results]))
            
            model_stats.append({
                'Modelo': model_name[:40],
                'F1': f"{avg_f1:.3f}",
                'NDCG': f"{avg_ndcg:.3f}",
                'P@1': f"{avg_p1:.3f}",
                'Emb.Time(s)': f"{avg_time:.1f}",
                'Configs': len(results)
            })
        
        df = pd.DataFrame(model_stats)
        df = df.sort_values('F1', ascending=False)
        print(df.to_string(index=False))
        print()
    
    def print_comparison_by_chunking(self):
        """Compara rendimiento por tamaño de chunk"""
        
        print(f"\n{'='*100}")
        print(f"📊 COMPARACIÓN POR ESTRATEGIA DE CHUNKING")
        print(f"{'='*100}\n")
        
        # Agrupar por chunk size
        by_chunk = defaultdict(list)
        for result in self.results:
            chunk_size = result['config']['chunk_size']
            by_chunk[chunk_size].append(result)
        
        # Calcular estadísticas por chunk size
        chunk_stats = []
        for chunk_size, results in sorted(by_chunk.items()):
            avg_f1 = float(np.mean([r['avg_metrics'].get('f1_score', 0) for r in results]))
            avg_ndcg = float(np.mean([r['avg_metrics'].get('ndcg_at_k', 0) for r in results]))
            avg_p1 = float(np.mean([r['avg_metrics'].get('precision_at_1', 0) for r in results]))
            avg_ctx = float(np.mean([r['avg_metrics'].get('avg_context_length', 0) for r in results]))
            
            chunk_stats.append({
                'Chunk Size': chunk_size,
                'F1': f"{avg_f1:.3f}",
                'NDCG': f"{avg_ndcg:.3f}",
                'P@1': f"{avg_p1:.3f}",
                'Avg Ctx': f"{avg_ctx:.0f}",
                'Configs': len(results)
            })
        
        df = pd.DataFrame(chunk_stats)
        df = df.sort_values('F1', ascending=False)
        print(df.to_string(index=False))
        print()
    
    def print_best_combinations(self):
        """Muestra las mejores combinaciones de modelo + chunking"""
        
        print(f"\n{'='*100}")
        print(f"🏆 TOP 10 MEJORES CONFIGURACIONES")
        print(f"{'='*100}\n")
        
        # Ordenar por F1 score
        sorted_results = sorted(
            self.results,
            key=lambda x: x['avg_metrics'].get('f1_score', 0),
            reverse=True
        )
        
        for i, result in enumerate(sorted_results[:10], 1):
            m = result['avg_metrics']
            config = result['config']
            
            # Emoji según posición
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            model_short = config['embedding_model'].split('/')[-1][:35]
            
            print(f"{emoji} {result['config_name']}")
            print(f"   Modelo: {model_short}")
            print(f"   F1: {m.get('f1_score', 0):.3f} | "
                  f"NDCG: {m.get('ndcg_at_k', 0):.3f} | "
                  f"P@1: {m.get('precision_at_1', 0):.3f} | "
                  f"MAP: {m.get('map_score', 0):.3f}")
            print(f"   Chunks: {config['chunk_size']} | "
                  f"K: {config['k']} | "
                  f"Ctx: {m.get('avg_context_length', 0):.0f} chars")
            print()
    
    def save_results(self, filename='outputs/results/rag_multi_model_evaluation.json'):
        """
        Guarda resultados en JSON
        
        Args:
            filename: Ruta del archivo de salida (relativa o absoluta)
        """
        
        # Convertir a ruta absoluta si es necesario
        if not os.path.isabs(filename):
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir.parent
            filename = str(project_root / filename)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados en: {filename}")
    
    def save_best_model(self, output_dir: str = "models"):
        """
        Guarda el mejor modelo RAG completo
        
        Incluye:
        - Índice FAISS
        - Configuración completa
        - Métricas de rendimiento
        - README con instrucciones de uso
        
        Args:
            output_dir: Directorio donde guardar el modelo
            
        Returns:
            Ruta del modelo guardado
        """
        
        if not self.results:
            print("❌ No hay resultados")
            return None
        
        # Encontrar mejor configuración por F1 score
        best = max(self.results, key=lambda x: x['avg_metrics'].get('f1_score', 0))
        
        print(f"\n{'='*100}")
        print(f"💾 GUARDANDO MEJOR MODELO RAG")
        print(f"{'='*100}")
        print(f"   Configuración: {best['config_name']}")
        print(f"   F1 Score: {best['avg_metrics'].get('f1_score', 0):.3f}")
        print(f"   NDCG: {best['avg_metrics'].get('ndcg_at_k', 0):.3f}")
        
        # Convertir a ruta absoluta
        if not os.path.isabs(output_dir):
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir.parent
            output_dir = str(project_root / output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "best_rag_model")
        os.makedirs(model_path, exist_ok=True)
        
        print(f"\n🔄 Recreando modelo...")
        
        # Recrear configuración
        config = RAGConfig()
        config.embedding_model = best['config']['embedding_model']
        config.chunk_size = best['config']['chunk_size']
        config.chunk_overlap = best['config']['chunk_overlap']
        config.k = best['config']['k']
        config.search_type = best['config']['search_type']
        
        # Crear RAG system
        rag = ConfigurableRAG(config)
        
        print(f"   📂 Cargando documentos...")
        documents = rag.load_documents(self.folder_path)
        
        print(f"   ✂️  Creando chunks...")
        chunks = rag.create_chunks(documents)
        
        print(f"   🔄 Creando embeddings...")
        rag.create_vectorstore(chunks)
        
        # Guardar FAISS index
        print(f"\n💾 Guardando FAISS...")
        faiss_path = os.path.join(model_path, "faiss_index")
        rag.vectorstore.save_local(faiss_path)
        print(f"   ✅ FAISS guardado")
        
        # Guardar configuración completa
        print(f"\n💾 Guardando configuración...")
        config_data = {
            'model_name': best['config_name'],
            'embedding_model': best['config']['embedding_model'],
            'chunk_size': best['config']['chunk_size'],
            'chunk_overlap': best['config']['chunk_overlap'],
            'k': best['config']['k'],
            'search_type': best['config']['search_type'],
            'fetch_k': config.fetch_k,
            'lambda_mult': config.lambda_mult,
            'normalize_embeddings': config.normalize_embeddings,
            'separators': config.separators,
            'performance_metrics': {
                'f1_score': float(best['avg_metrics'].get('f1_score', 0)),
                'ndcg': float(best['avg_metrics'].get('ndcg_at_k', 0)),
                'precision_at_1': float(best['avg_metrics'].get('precision_at_1', 0)),
                'precision_at_3': float(best['avg_metrics'].get('precision_at_3', 0)),
                'precision_at_5': float(best['avg_metrics'].get('precision_at_5', 0)),
                'map_score': float(best['avg_metrics'].get('map_score', 0)),
                'mrr': float(best['avg_metrics'].get('mrr', 0)),
                'coverage': float(best['avg_metrics'].get('coverage', 0)),
                'avg_retrieval_time': float(best['avg_metrics'].get('retrieval_time', 0)),
                'avg_context_length': float(best['avg_metrics'].get('avg_context_length', 0)),
            },
            'model_info': best['config'].get('model_info', {}),
            'total_chunks': int(best['avg_metrics'].get('total_chunks_indexed', 0)),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': self.test_dataset.get_statistics()
        }
        
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Configuración guardada")
        
        # Crear README
        print(f"\n💾 Creando README...")
        readme_content = f"""# Best RAG Model - F1 Regulations 2025

## Model Information

**Configuration**: {best['config_name']}  
**Created**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics

- **F1 Score**: {best['avg_metrics'].get('f1_score', 0):.3f}
- **NDCG@K**: {best['avg_metrics'].get('ndcg_at_k', 0):.3f}
- **Precision@1**: {best['avg_metrics'].get('precision_at_1', 0):.3f}
- **MAP**: {best['avg_metrics'].get('map_score', 0):.3f}
- **Coverage**: {best['avg_metrics'].get('coverage', 0):.3f}

## Configuration

- **Embedding Model**: {best['config']['embedding_model']}
- **Chunk Size**: {best['config']['chunk_size']}
- **K**: {best['config']['k']}
- **Total Chunks**: {int(best['avg_metrics'].get('total_chunks_indexed', 0))}

## Usage

```python
from chatbot import F1RAGChatbot

chatbot = F1RAGChatbot(model_path="models/best_rag_model")
result = chatbot.query("How many power units can a driver use?")
```
"""
        
        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"   ✅ README creado")
        
        print(f"\n{'='*100}")
        print(f"✅ MODELO GUARDADO")
        print(f"{'='*100}")
        print(f"📁 Ubicación: {model_path}")
        print(f"{'='*100}\n")
        
        return model_path
    
    def generate_comparison_report(self):
        """Genera reporte final con resumen de resultados"""
        
        print(f"\n{'='*100}")
        print(f"📋 REPORTE FINAL")
        print(f"{'='*100}\n")
        
        if not self.results:
            print("No hay resultados")
            return
        
        best = max(self.results, key=lambda x: x['avg_metrics'].get('f1_score', 0))
        
        print(f"🏆 MEJOR CONFIGURACIÓN:")
        print(f"   {best['config_name']}")
        print(f"   F1: {best['avg_metrics'].get('f1_score', 0):.3f}")
        print(f"   NDCG: {best['avg_metrics'].get('ndcg_at_k', 0):.3f}")
        
        print(f"\n{'='*100}\n")

# ====================================
# FUNCIONES AUXILIARES
# ====================================

def print_dataset_info():
    """Muestra información del dataset de evaluación"""
    
    try:
        dataset = F1TestDataset()
        stats = dataset.get_statistics()
        
        print(f"\n{'='*100}")
        print(f"📋 DATASET DE EVALUACIÓN")
        print(f"{'='*100}")
        print(f"   Nombre: {stats['dataset_info'].get('name', 'N/A')}")
        print(f"   Versión: {stats['dataset_info'].get('version', 'N/A')}")
        print(f"   Idioma: {stats['dataset_info'].get('language', 'N/A')}")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"\n   Por dificultad:")
        for diff, count in stats['by_difficulty'].items():
            print(f"      {diff}: {count}")
        print(f"\n   Por categoría:")
        for cat, count in sorted(stats['by_category'].items()):
            print(f"      {cat}: {count}")
        print(f"{'='*100}\n")
        
        # Validar dataset
        dataset.validate_dataset()
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print(f"\n💡 Crea data/test_dataset.json con tus queries de prueba")
        sys.exit(1)

# ====================================
# MAIN
# ====================================

def main():
    """
    Función principal que ejecuta el experimento completo
    
    1. Carga el dataset de evaluación
    2. Muestra catálogo de modelos
    3. Define configuraciones a probar
    4. Ejecuta evaluación multi-modelo
    5. Genera reportes y guarda mejor modelo
    """
    
    print("\n" + "="*100)
    print("🏎️  EVALUADOR MULTI-MODELO RAG - F1 2025")
    print("   Dataset cargado desde JSON")
    print("="*100 + "\n")
    
    # Mostrar información del dataset
    print_dataset_info()
    
    # Mostrar catálogo de modelos disponibles
    EmbeddingModels.print_catalog()
    
    # Determinar ruta a documentos procesados
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    folder_path = str(project_root / "data" / "processed")
    
    print(f"🔧 Directorio de documentos: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: {folder_path} no encontrado")
        print(f"   Por favor, ejecuta primero process_f1_documents.py")
        sys.exit(1)
    
    print("\n🔧 Configurando experimento...\n")
    
    # Configurar modelos a probar
    embedding_models = [
        'multilingual-mini',  # Rápido, multilingüe
        'all-minilm',         # Muy rápido, inglés
        'bge-small',          # Excelente rendimiento, inglés
    ]
    
    # Configurar estrategias de chunking a probar
    chunking_strategies = [
        {'chunk_size': 500, 'chunk_overlap': 100, 'k': 3},    # Chunks pequeños
        {'chunk_size': 1000, 'chunk_overlap': 200, 'k': 5},   # Chunks medianos, más documentos
        {'chunk_size': 2000, 'chunk_overlap': 400, 'k': 3},   # Chunks grandes
    ]
    
    print(f"📊 Modelos a probar: {', '.join(embedding_models)}")
    print(f"📏 Tamaños de chunks: {[s['chunk_size'] for s in chunking_strategies]}")
    print(f"🔢 Total configuraciones: {len(embedding_models) * len(chunking_strategies)}\n")
    
    # Crear comparador y ejecutar experimento
    comparator = MultiModelRAGComparator(folder_path)
    results = comparator.compare_models_and_strategies(
        embedding_models=embedding_models,
        chunking_strategies=chunking_strategies
    )
    
    # Generar reportes
    comparator.generate_comparison_report()
    
    # Guardar mejor modelo
    model_path = comparator.save_best_model(output_dir="models")
    
    print("\n✅ EVALUACIÓN COMPLETA!")
    print(f"🤖 Modelo guardado en: {model_path}")
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()