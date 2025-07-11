#!/usr/bin/env python3

import sys
import os
import logging

# Adicionar diretórios ao path
sys.path.insert(0, "src/core/core_logic")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_retrieval():
    """Testa o sistema de recuperação com debug detalhado."""
    try:
        from rag_infra.src.core.rag_retriever import RAGRetriever
        
        logger.info("Inicializando RAGRetriever...")
        retriever = RAGRetriever()
        
        logger.info("Carregando índice...")
        retriever.load_index()
        
        logger.info("Fazendo busca por 'arquitetura do sistema'...")
        results = retriever.search('arquitetura do sistema', top_k=3)
        
        logger.info(f"Resultados encontrados: {len(results) if results else 0}")
        
        if results:
            for i, result in enumerate(results, 1):
                logger.info(f"Resultado {i}: {result.metadata.get('document_name', 'N/A')} (score: {result.score:.3f})")
                logger.info(f"  Conteúdo: {result.content[:100]}...")
        else:
            logger.warning("Nenhum resultado encontrado")
            
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retrieval()
    print(f"/nTeste {'PASSOU' if success else 'FALHOU'}")