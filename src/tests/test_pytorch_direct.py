#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste direto do PyTorch GPU Retriever
"""

import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Adicionar o diretório do projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pytorch_retriever():
    """Testa o PyTorch GPU Retriever diretamente"""
    
    try:
        logger.info("=== Teste Direto PyTorch GPU Retriever ===")
        
        # Importar o retriever
        logger.info("[EMOJI] Importando PyTorchGPURetriever...")
        from rag_infra.src.core.core_logic.pytorch_gpu_retriever import PyTorchGPURetriever
        logger.info("[OK] Import bem-sucedido")
        
        # Criar instância
        logger.info("[EMOJI] Criando instância do retriever...")
        retriever = PyTorchGPURetriever()
        logger.info("[OK] Instância criada")
        
        # Inicializar
        logger.info("[START] Inicializando retriever...")
        init_result = retriever.initialize()
        logger.info(f"Resultado da inicialização: {init_result}")
        
        if not init_result:
            logger.error("[ERROR] Falha na inicialização")
            return False
        
        # Carregar índice
        logger.info("[EMOJI] Carregando índice...")
        load_result = retriever.load_index()
        logger.info(f"Resultado do carregamento: {load_result}")
        
        if not load_result:
            logger.error("[ERROR] Falha no carregamento do índice")
            return False
        
        # Verificar estado
        logger.info(f"[EMOJI] Estado do retriever:")
        logger.info(f"  - is_loaded: {retriever.is_loaded}")
        logger.info(f"  - device: {retriever.device}")
        logger.info(f"  - documentos: {len(retriever.documents) if retriever.documents else 0}")
        logger.info(f"  - metadados: {len(retriever.metadata) if retriever.metadata else 0}")
        
        if hasattr(retriever, 'embeddings_tensor') and retriever.embeddings_tensor is not None:
            logger.info(f"  - embeddings shape: {retriever.embeddings_tensor.shape}")
        else:
            logger.warning("  - embeddings: não carregados")
        
        # Teste de busca simples
        if retriever.is_loaded:
            logger.info("[SEARCH] Testando busca simples...")
            results = retriever.search("FastAPI", top_k=3)
            logger.info(f"Resultados encontrados: {len(results)}")
            
            for i, result in enumerate(results[:2]):
                logger.info(f"  {i+1}. Score: {result.score:.4f} - {result.content[:100]}...")
        
        logger.info("[OK] Teste concluído com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Erro no teste: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_pytorch_retriever()
    if success:
        print("\n=== Resultado Final: SUCESSO ===")
    else:
        print("\n=== Resultado Final: FALHA ===")