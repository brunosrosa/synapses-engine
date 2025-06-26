# -*- coding: utf-8 -*-
"""
Teste simples de busca com threshold 0.0
"""

import os
import sys
from pathlib import Path

# Adicionar o diretório src ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.rag_retriever import RAGRetriever

def test_search_with_zero_threshold():
    """
    Testa busca com threshold 0.0
    """
    print("=== Teste de Busca com Threshold 0.0 ===")
    
    # Inicializar retriever
    retriever = RAGRetriever()
    
    try:
        retriever.initialize()
        print("[OK] RAGRetriever inicializado")
    except Exception as e:
        print(f"[ERROR] Erro na inicialização: {e}")
        raise e
    
    try:
        retriever.load_index()
        print(f"[OK] Índice carregado: {len(retriever.documents)} documentos")
    except Exception as e:
        print(f"[ERROR] Erro ao carregar índice: {e}")
        raise e
    
    # Testar busca com threshold 0.0
    query = "FastAPI"
    print(f"\nTestando busca: '{query}' com min_score=0.0")
    
    try:
        results = retriever.search(query, top_k=5, min_score=0.0)
        print(f"Resultados encontrados: {len(results)}")
        
        if results:
            for i, result in enumerate(results):
                print(f"\nResultado {i+1}:")
                print(f"  Score: {result.score:.4f}")
                print(f"  Conteúdo: {result.content[:100]}...")
                print(f"  Metadata: {result.metadata}")
        else:
            print("[ERROR] Nenhum resultado encontrado mesmo com threshold 0.0")
            
    except Exception as e:
        print(f"[ERROR] Erro na busca: {e}")
        import traceback
        traceback.print_exc()

