"""
Ejemplo de cómo importar y usar el chatbot desde scripts/

Este script demuestra la forma correcta de importar módulos 
desde el directorio src/ cuando se ejecuta desde scripts/
"""

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

# Ahora podemos importar desde src/ usando import absoluto
from src.chatbot import F1RAGChatbot

# ====================================
# EJEMPLO DE USO
# ====================================

def test_chatbot():
    """Prueba básica del chatbot con nuevas funcionalidades"""
    
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
    
    # Obtener respuesta con formato estructurado
    result = chatbot.query(
        test_question,
        max_results=3,
        max_total_chars=800,
        generate_answer=True  # Generar respuesta formateada
    )
    
    # Mostrar respuesta generada
    print("="*80)
    print("📄 RESPUESTA GENERADA (Formatted Answer):")
    print("="*80)
    if 'answer' in result:
        print(result['answer'])
    else:
        print(result['context'])
    
    # Mostrar contexto raw para comparación
    print("\n" + "="*80)
    print("📚 CONTEXTO RAW (Para debugging):")
    print("="*80)
    print(result['context'])
    
    print("\n" + "="*80)
    print(f"📊 Fuentes utilizadas: {result['num_sources']}")
    print("="*80)
    
    # Mostrar fuentes detalladas
    if 'sources' in result:
        print("\n📚 Detalles de las fuentes:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n   {i}. Artículo {source['article_number']}")
            print(f"      Archivo: {source['source_file']}")
            print(f"      Categoría: {source['category']}")
            print(f"      Extracto: {source['content'][:100]}...")
    
    print("\n" + "="*80)
    print("💡 COMPARACIÓN:")
    print("   - 'Respuesta Generada' = Formato estructurado y profesional")
    print("   - 'Contexto Raw' = Información directa de los documentos")
    print("="*80 + "\n")

def test_multiple_queries():
    """Prueba con múltiples consultas mostrando respuestas generadas"""
    
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
        print(f"\n{'='*80}")
        print(f"[{i}/{len(test_questions)}] {question}")
        print("="*80)
        
        result = chatbot.query(
            question,
            max_results=2,
            max_total_chars=600,
            generate_answer=True  # Usar respuestas generadas
        )
        
        # Mostrar respuesta generada
        if 'answer' in result:
            print("\n📄 Respuesta:\n")
            print(result['answer'])
        else:
            print("\n📄 Contexto:\n")
            print(result['context'])
        
        print(f"\n📊 Fuentes: {result['num_sources']}")
    
    print("\n" + "="*80 + "\n")

def test_comparison_modes():
    """Compara respuestas generadas vs contexto raw"""
    
    print("\n" + "="*80)
    print("🧪 COMPARACIÓN: RESPUESTA GENERADA VS CONTEXTO RAW")
    print("="*80 + "\n")
    
    chatbot = F1RAGChatbot()
    
    test_question = "If race is not completed to 70% indicate points and positions"
    
    print(f"Pregunta: {test_question}\n")
    
    # Obtener ambas versiones
    result_formatted = chatbot.query(
        test_question,
        max_results=3,
        generate_answer=True
    )
    
    result_raw = chatbot.query(
        test_question,
        max_results=3,
        generate_answer=False
    )
    
    # Mostrar modo formateado
    print("="*80)
    print("📄 MODO 1: RESPUESTA GENERADA (Usuario final)")
    print("="*80)
    print(result_formatted.get('answer', 'N/A'))
    
    # Mostrar modo raw
    print("\n" + "="*80)
    print("📄 MODO 2: CONTEXTO RAW (Debugging/Desarrollo)")
    print("="*80)
    print(result_raw['context'])
    
    print("\n" + "="*80)
    print("💡 ANÁLISIS:")
    print("   - Modo 1: Profesional, estructurado, listo para usuario")
    print("   - Modo 2: Directo de documentos, útil para verificar fuentes")
    print("="*80 + "\n")

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
        choices=['test', 'multiple', 'compare', 'interactive'],
        default='test',
        help='Modo de ejecución (default: test)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'test':
            test_chatbot()
        elif args.mode == 'multiple':
            test_multiple_queries()
        elif args.mode == 'compare':
            test_comparison_modes()
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