import os
import sys
from pathlib import Path

# ====================================
# CONFIGURACIÓN DE RUTAS
# ====================================

# Obtener directorio del script actual
script_dir = Path(__file__).parent.absolute()

# Subir un nivel para llegar a la raíz del proyecto
project_root = script_dir.parent

# Agregar la raíz del proyecto al path de Python
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Importar desde src/ usando import absoluto
from src.chatbot import F1RAGChatbot

# ====================================
# EJEMPLO DE USO
# ====================================

def test_chatbot():
    """Prueba básica del chatbot"""
    
    print("\n" + "="*80)
    print("🧪 TEST DEL CHATBOT F1")
    print("="*80 + "\n")
    
    # Inicializar chatbot
    # La ruta se resolverá automáticamente desde project_root
    print("📂 Cargando modelo...")
    chatbot = F1RAGChatbot()
    
    # Mostrar información del modelo
    print("\n📊 Información del modelo:")
    info = chatbot.get_info()
    for key, value in info.items():
        if key != 'performance_metrics':
            print(f"   {key}: {value}")
    
    # Métricas de rendimiento
    if 'performance_metrics' in info:
        print("\n   Métricas de rendimiento:")
        for key, value in info['performance_metrics'].items():
            if isinstance(value, float):
                print(f"      {key}: {value:.3f}")
            else:
                print(f"      {key}: {value}")
    
    # Realizar consulta de prueba
    print("\n" + "="*80)
    print("🔍 CONSULTA DE PRUEBA")
    print("="*80 + "\n")
    
    test_question = "How many power units can a driver use in a season?"
    
    print(f"Pregunta: {test_question}\n")
    print("Buscando en las regulaciones...\n")
    
    result = chatbot.query(
        test_question,
        max_results=3,
        max_total_chars=800
    )
    
    print("📄 Contexto encontrado:")
    print("-" * 80)
    print(result['context'])
    print("-" * 80)
    print(f"\n📊 Fuentes utilizadas: {result['num_sources']}")
    
    # Mostrar fuentes detalladas
    if 'sources' in result:
        print("\n📚 Detalles de las fuentes:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n   {i}. Artículo {source['article_number']}")
            print(f"      Archivo: {source['source_file']}")
            print(f"      Categoría: {source['category']}")
            print(f"      Extracto: {source['content'][:100]}...")
    
    print("\n" + "="*80 + "\n")

def test_multiple_queries():
    """Prueba con múltiples consultas"""
    
    print("\n" + "="*80)
    print("🧪 TEST MÚLTIPLES CONSULTAS")
    print("="*80 + "\n")
    
    # Inicializar chatbot una sola vez
    chatbot = F1RAGChatbot()
    
    # Lista de preguntas de prueba
    test_questions = [
        "What are the minimum weight requirements?",
        "How many races can be held in a season?",
        "What happens if a race is not completed?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] {question}")
        print("-" * 80)
        
        result = chatbot.query(
            question,
            max_results=2,
            max_total_chars=400
        )
        
        print(result['context'])
        print(f"\nFuentes: {result['num_sources']}")
        print("=" * 80)

def interactive_session():
    """Sesión interactiva con el chatbot"""
    
    chatbot = F1RAGChatbot()
    chatbot.interactive_mode()

# ====================================
# MAIN
# ====================================

def main():
    """Función principal"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Test del chatbot F1')
    parser.add_argument(
        '--mode',
        choices=['test', 'multiple', 'interactive'],
        default='test',
        help='Modo de ejecución (default: test)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'test':
            test_chatbot()
        elif args.mode == 'multiple':
            test_multiple_queries()
        elif args.mode == 'interactive':
            interactive_session()
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Asegúrate de que:")
        print("   1. Has ejecutado evaluate_rag_strategies_with_models.py")
        print("   2. Existe el directorio models/best_rag_model/")
        print("   3. El modelo contiene config.json y faiss_index/\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()