#!/usr/bin/env python3
"""
Teste básico de imports do sistema RAG
"""

import sys
from pathlib import Path

# Adicionar core_logic ao path

project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root / "src"))

print("[SEARCH] Testando imports básicos do sistema RAG...")

try:
    from constants import *
    print("[OK] Constants importado com sucesso")
except ImportError as e:
    print(f"[ERROR] Erro ao importar constants: {e}")
    raise

try:
    from core.embedding_model import EmbeddingModelManager as EmbeddingModel
    print("[OK] EmbeddingModel importado com sucesso")
except ImportError as e:
    print(f"[ERROR] Erro ao importar embedding_model: {e}")
    raise

try:
    from rag_retriever import RAGRetriever
    print("[OK] RAGRetriever importado com sucesso")
except ImportError as e:
    print(f"[ERROR] Erro ao importar rag_retriever: {e}")
    raise

try:
    from rag_indexer import RAGIndexer
    print("[OK] RAGIndexer importado com sucesso")
except ImportError as e:
    print(f"[ERROR] Erro ao importar rag_indexer: {e}")
    raise

print("\n[EMOJI] Teste de imports básicos concluído!")