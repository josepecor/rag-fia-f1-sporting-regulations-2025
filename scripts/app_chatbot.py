
import streamlit as st
import os
import sys
from pathlib import Path

# Configurar rutas para importar desde src/
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.chatbot import F1RAGChatbot
except ImportError:
    st.error("❌ No se pudo importar chatbot.py desde src/")
    st.stop()

# ====================================
# CONFIGURACIÓN
# ====================================

st.set_page_config(
    page_title="F1 Regulations Chatbot",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title-container {
        background: linear-gradient(135deg, #e10600 0%, #ff1e1e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .title-container h1 { color: white; margin: 0; font-size: 2.5rem; }
    .title-container p { color: #f0f0f0; margin: 0.5rem 0 0 0; }
    .user-message {
        background-color: #1e3a5f;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4a90e2;
    }
    .assistant-message {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #e10600;
    }
    .welcome-message {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e10600;
    }
</style>
""", unsafe_allow_html=True)

# ====================================
# INICIALIZACIÓN
# ====================================

def init_session_state():
    """Inicializa el estado de la sesión"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chatbot_loaded' not in st.session_state:
        st.session_state.chatbot_loaded = False
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None
    if 'model_load_attempted' not in st.session_state:
        st.session_state.model_load_attempted = False

@st.cache_resource
def load_chatbot(model_path: str):
    """
    Carga el modelo RAG (con caché para no recargar)
    
    Args:
        model_path: Ruta al modelo
        
    Returns:
        Tupla (chatbot, error_message)
    """
    try:
        chatbot = F1RAGChatbot(model_path=model_path)
        return chatbot, None
    except Exception as e:
        return None, str(e)

# ====================================
# MENSAJE DE BIENVENIDA
# ====================================

def show_welcome_message():
    """Muestra el mensaje de bienvenida inicial"""
    
    # Usar un contenedor con estilo
    with st.container():
        # Título principal
        st.markdown(
            '<h2 style="color: #e10600; text-align: center;">🏎️ Welcome to F1 Regulations Assistant!</h2>',
            unsafe_allow_html=True
        )
        
        # Descripción
        st.markdown(
            """
            <p style="font-size: 1.1rem; color: #e0e0e0; text-align: center;">
                I'm your specialized assistant for the <strong>FIA Formula 1 Sporting Regulations - 2025 Season</strong>.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Sección de ayuda - usar markdown nativo
        st.markdown("### I can help you find information about:")
        
        st.markdown("""
        - ⚙️ **Technical regulations**: Power units, tyres, restrictions
        - 🏁 **Race procedures**: Starts, safety car, flags  
        - 📋 **Classification & points**: Scoring system, sprints
        - ⚖️ **Penalties**: Infractions and sanctions
        - 🔧 **Team operations**: Parc fermé, scrutineering, pit lane
        """)
        
        st.markdown("---")
        
        # Instrucción final
        st.info("💡 Ask a question below or select an example from the sidebar to get started!")

# ====================================
# VISUALIZACIÓN DE MENSAJES
# ====================================

def display_chat_message(role: str, content: str, sources: list = None):
    """
    Muestra un mensaje del chat
    
    Args:
        role: 'user' o 'assistant'
        content: Contenido del mensaje
        sources: Lista de fuentes (opcional)
    """
    if role == "user":
        # Mensaje del usuario con estilo azul
        st.markdown(
            f"""
            <div style="background-color: #1e3a5f; padding: 1rem; border-radius: 10px; 
                        margin: 0.5rem 0; border-left: 4px solid #4a90e2;">
                <strong style="color: #4a90e2;">🧑 You:</strong><br/>
                <span style="color: #e0e0e0;">{content}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    elif role == "assistant":
        # Mensaje del asistente con estilo rojo
        st.markdown(
            f"""
            <div style="background-color: #1a1a1a; padding: 1rem; border-radius: 10px; 
                        margin: 0.5rem 0; border-left: 4px solid #e10600;">
                <strong style="color: #e10600;">🤖 Assistant:</strong><br/>
                <span style="color: #e0e0e0;">{content}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Mostrar fuentes si existen
        if sources and len(sources) > 0:
            st.markdown("---")
            st.markdown("### 📚 Sources:")
            
            for i, source in enumerate(sources, 1):
                article_num = source.get('article_number', 'N/A')
                content_preview = source.get('content', '')[:300] + "..."
                
                with st.expander(f"📄 Article {article_num} - Source {i}"):
                    st.markdown(f"**Category:** {source.get('category', 'N/A')}")
                    st.markdown(f"**File:** {source.get('source_file', 'N/A')}")
                    st.markdown("**Preview:**")
                    st.text(content_preview)

# ====================================
# PROCESAMIENTO DE PREGUNTAS
# ====================================

def process_question(question: str):
    """
    Procesa una pregunta y genera la respuesta
    
    Args:
        question: Pregunta del usuario
    """
    
    if not st.session_state.chatbot_loaded or not st.session_state.chatbot:
        st.error("⚠️ El modelo no está cargado")
        return
    
    # Agregar pregunta del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    
    # Generar respuesta
    try:
        result = st.session_state.chatbot.query(
            question, 
            return_sources=True,
            max_results=3,
            max_total_chars=800
        )
        
        response_text = result['context']
        sources = result.get('sources', [])
        
        # Agregar respuesta del asistente
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources
        })
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        # Agregar mensaje de error como respuesta
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, I encountered an error: {str(e)}",
            "sources": []
        })

# ====================================
# PREGUNTAS DE EJEMPLO
# ====================================

EXAMPLE_QUESTIONS = {
    "⚙️ Technical": [
        "How many power units can a driver use?",
        "What are the tyre allocation rules?",
        "What are the parc fermé regulations?",
    ],
    "🏁 Procedures": [
        "What is the race starting procedure?",
        "When is the safety car deployed?",
        "What happens if race is suspended?",
    ],
    "📋 Points": [
        "If race is not completed to 70% indicate points and positions",
        "How does the sprint qualifying work?",
        "What is the points system?",
    ],
    "⚖️ Penalties": [
        "What are the different types of penalties?",
        "What happens if a driver receives a grid penalty?",
        "What are the pit lane speed limits?",
    ],
}

# ====================================
# MAIN
# ====================================

def main():
    """Función principal de la aplicación"""
    
    init_session_state()
    
    # Header
    st.markdown(
        """
        <div class="title-container">
            <h1>🏎️ F1 Regulations Chatbot</h1>
            <p>FIA Formula 1 Sporting Regulations 2025</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏎️ F1 Regulations")
        st.markdown("Ask any question about the 2025 F1 regulations")
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        
        for category, questions in EXAMPLE_QUESTIONS.items():
            with st.expander(category):
                for question in questions:
                    if st.button(question, key=f"ex_{question}", use_container_width=True):
                        if st.session_state.chatbot_loaded:
                            st.session_state.show_welcome = False
                            st.session_state.pending_question = question
                            st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.show_welcome = True
            st.session_state.pending_question = None
            st.rerun()
    
    # Carga automática del modelo al iniciar
    if not st.session_state.model_load_attempted:
        st.session_state.model_load_attempted = True
        
        with st.spinner("🔄 Loading F1 regulations model..."):
            # Ruta predeterminada al modelo
            model_path = "models/best_rag_model"
            
            chatbot, error = load_chatbot(model_path)
            
            if error:
                st.error(f"❌ Error loading model: {error}")
                st.error("Please make sure the model exists at: models/best_rag_model/")
                st.session_state.chatbot_loaded = False
            else:
                st.session_state.chatbot = chatbot
                st.session_state.chatbot_loaded = True
                st.success("✅ Model loaded successfully!")
    
    # Si el modelo no está cargado, mostrar instrucciones
    if not st.session_state.chatbot_loaded:
        st.warning("⚠️ Model could not be loaded")
        st.info("Please ensure the model exists at: **models/best_rag_model/**")
        
        st.markdown("""
        ### 📋 How to generate the model:
        
        1. **Process documents**: Run `python scripts/process_f1_documents.py`
        2. **Evaluate models**: Run `python scripts/evaluate_rag_strategies_with_models.py`
        3. **Restart the app**: The model should load automatically
        """)
        
        return
    
    # Procesar pregunta pendiente (de botones de ejemplo)
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        
        with st.spinner("🔍 Searching regulations..."):
            process_question(question)
        
        st.rerun()
    
    # Mostrar mensaje de bienvenida si no hay mensajes
    if st.session_state.show_welcome and len(st.session_state.messages) == 0:
        show_welcome_message()
    
    # Mostrar historial de chat
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("sources")
            )
    
    # Input del usuario
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        st.session_state.show_welcome = False
        
        with st.spinner("🔍 Searching regulations..."):
            process_question(user_input)
        
        st.rerun()

if __name__ == "__main__":
    main()