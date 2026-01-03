# src/__init__.py
"""
Paquetes locales
"""

__version__ = '0.1.0'

# Importar funciones principales para acceso directo
from .chatbot import F1RAGChatbot
__all__ = [
    'F1RAGChatbot'
]