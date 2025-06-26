# -*- coding: utf-8 -*-
"""
Módulo de Embedding para o Sistema RAG do Recoloca.ai

Este módulo gerencia o modelo de embedding BGE-M3, incluindo carregamento,
configuração de GPU/CPU e geração de embeddings para documentos e consultas.

Autor: @AgenteM_DevFastAPI
Versão: 1.0
Data: Junho 2025
"""

import logging
import time
from typing import List, Optional, Union, Dict, Any
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config_manager import RAGConfig
try:
    from .torch_utils import is_cuda_available, safe_cuda_empty_cache
except ImportError:
    from torch_utils import is_cuda_available, safe_cuda_empty_cache

logger = logging.getLogger(__name__)

class EmbeddingModelManager:
    """
    Gerenciador do modelo de embedding BGE-M3.
    
    Esta classe encapsula toda a lógica de carregamento, configuração
    e uso do modelo de embedding, incluindo otimizações para GPU.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Inicializa o gerenciador de embedding.
        
        Args:
            config: Objeto de configuração RAGConfig.
        """
        self.config = config
        self.model: Optional[Union[SentenceTransformer, HuggingFaceEmbeddings]] = None
        self.device = self._detect_device()
        self.model_name = self.config.embedding.name
        self.is_loaded = False
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Configurar logging usando RAGConfig
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.config.logs_dir / "embedding_model.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"EmbeddingModelManager inicializado. Dispositivo: {self.device}")
    
    def _detect_device(self) -> str:
        """
        Detecta o melhor dispositivo disponível (GPU/CPU).
        
        Returns:
            str: 'cuda' ou 'cpu'
        """
        if self.config.force_cpu:
            logger.info("Forçando uso de CPU")
            return "cpu"
        
        if is_cuda_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info("GPU disponível para operações de embedding")
            logger.info(f"GPU detectada: {gpu_name} (Total: {gpu_count})")
            return "cuda"
        else:
            logger.warning("GPU não disponível, usando CPU para operações de embedding")
            return "cpu"
    
    def load_model(self) -> bool:
        """
        Carrega o modelo de embedding.
        
        Returns:
            bool: True se carregado com sucesso, False caso contrário
        """
        try:
            logger.info(f"Carregando modelo de embedding: {self.model_name}")
            start_time = time.time()
            
            # Por padrão, usar SentenceTransformer
            if False:  # Desabilitado por enquanto
                self.model = self._load_langchain_model()
            else:
                self.model = self._load_sentence_transformer()
            
            load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info("Modelo de embedding carregado com sucesso")
            logger.info(f"Tempo de carregamento: {load_time:.2f}s")
            
            # Teste rápido do modelo
            self._test_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            self.is_loaded = False
            return False
    
    def _load_langchain_model(self) -> HuggingFaceEmbeddings:
        """
        Carrega o modelo usando HuggingFaceEmbeddings (LangChain).
        
        Returns:
            HuggingFaceEmbeddings: Modelo carregado
        """
        model_kwargs = {}
        model_kwargs["device"] = self.device
        
        encode_kwargs = {}
        
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def _load_sentence_transformer(self) -> SentenceTransformer:
        """
        Carrega o modelo usando SentenceTransformer diretamente.
        
        Returns:
            SentenceTransformer: Modelo carregado
        """
        sentence_transformer_kwargs = {}
        encode_kwargs = {}

        model = SentenceTransformer(
            self.model_name,
            device=self.device,
            **sentence_transformer_kwargs
        )
        
        # Configurar para normalizar embeddings se especificado em encode_kwargs
        if encode_kwargs.get("normalize_embeddings", False):
            model.encode = lambda texts, **kwargs: model.encode(
                texts, 
                normalize_embeddings=True,
                **kwargs
            )
        
        return model
    
    def _test_model(self) -> None:
        """
        Testa o modelo com uma consulta simples.
        """
        try:
            test_text = "Teste de funcionamento do modelo de embedding"
            embedding = self.embed_query(test_text)
            
            if embedding is not None and len(embedding) > 0:
                logger.info(f"[OK] Teste do modelo bem-sucedido. Dimensão: {len(embedding)}")
            else:
                logger.warning("[WARNING] Teste do modelo retornou embedding vazio")
                
        except Exception as e:
            logger.error(f"[ERROR] Falha no teste do modelo: {str(e)}")
    
    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Gera embedding para uma consulta.
        
        Args:
            query: Texto da consulta
            
        Returns:
            np.ndarray: Embedding da consulta ou None se erro
        """
        if not self.is_loaded:
            logger.error("Modelo não carregado. Chame load_model() primeiro.")
            return None
        
        # Verificar cache
        if query in self._embedding_cache:
            logger.debug(f"Embedding encontrado no cache para: {query[:50]}...")
            return self._embedding_cache[query]
        
        try:
            start_time = time.time()
            
            if isinstance(self.model, HuggingFaceEmbeddings):
                embedding = self.model.embed_query(query)
            else:  # SentenceTransformer
                embedding = self.model.encode([query])[0]
            
            # Converter para numpy array se necessário
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Normalizar se não estiver normalizado
            if not np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-6):
                embedding = embedding / np.linalg.norm(embedding)
            
            # Cache do resultado
            self._embedding_cache[query] = embedding
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Embedding gerado em {elapsed_time:.3f}s para: {query[:50]}...")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding para consulta: {str(e)}")
            return None
    
    def embed_documents(self, documents: List[str], batch_size: int = 32) -> Optional[List[np.ndarray]]:
        """
        Gera embeddings para uma lista de documentos.
        
        Args:
            documents: Lista de textos dos documentos
            batch_size: Tamanho do batch para processamento
            
        Returns:
            List[np.ndarray]: Lista de embeddings ou None se erro
        """
        if not self.is_loaded:
            logger.error("Modelo não carregado. Chame load_model() primeiro.")
            return None
        
        if not documents:
            logger.warning("Lista de documentos vazia")
            return []
        
        try:
            start_time = time.time()
            embeddings = []
            
            # Processar em batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                if isinstance(self.model, HuggingFaceEmbeddings):
                    batch_embeddings = self.model.embed_documents(batch)
                else:  # SentenceTransformer
                    batch_embeddings = self.model.encode(batch)
                
                # Converter para numpy arrays e normalizar
                for emb in batch_embeddings:
                    if not isinstance(emb, np.ndarray):
                        emb = np.array(emb)
                    
                    # Normalizar se necessário
                    if not np.allclose(np.linalg.norm(emb), 1.0, atol=1e-6):
                        emb = emb / np.linalg.norm(emb)
                    
                    embeddings.append(emb)
                
                logger.debug(f"Processado batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Embeddings gerados para {len(documents)} documentos em {elapsed_time:.2f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings para documentos: {str(e)}")
            return None
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Retorna a dimensão dos embeddings do modelo.
        
        Returns:
            int: Dimensão dos embeddings ou None se modelo não carregado
        """
        if not self.is_loaded:
            return None
        
        try:
            # Gerar embedding de teste para descobrir a dimensão
            test_embedding = self.embed_query("teste")
            return len(test_embedding) if test_embedding is not None else None
            
        except Exception as e:
            logger.error(f"Erro ao obter dimensão do embedding: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """
        Limpa o cache de embeddings.
        """
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info(f"Cache limpo. {cache_size} embeddings removidos.")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o cache.
        
        Returns:
            Dict: Informações do cache
        """
        return {
            "cache_size": len(self._embedding_cache),
            "model_loaded": self.is_loaded,
            "device": self.device,
            "model_name": self.model_name
        }
    
    def unload_model(self) -> None:
        """
        Descarrega o modelo da memória.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
                self.is_loaded = False
                
                # Limpar cache CUDA se usando GPU
                if self.device == "cuda":
                    safe_cuda_empty_cache()
                
                logger.info("Modelo descarregado da memória")
        except (AttributeError, RuntimeError):
            # Ignorar erros durante cleanup - módulos podem já ter sido descarregados
            pass
    
    def __del__(self):
        """
        Destrutor para limpeza automática.
        """
        try:
            self.unload_model()
        except (AttributeError, RuntimeError):
            # Ignorar erros durante cleanup - módulos podem já ter sido descarregados
            pass

def initialize_embedding_model(config: RAGConfig) -> EmbeddingModelManager:
    """
    Inicializa e retorna uma nova instância do EmbeddingModelManager.
    
    Args:
        config: Objeto de configuração RAGConfig.
        
    Returns:
        EmbeddingModelManager: Instância do gerenciador de embedding.
    """
    manager = EmbeddingModelManager(config=config)
    if not manager.load_model():
        raise RuntimeError("Falha ao carregar o modelo de embedding.")
    return manager


def get_embedding_manager(config: RAGConfig) -> EmbeddingModelManager:
    """
    Função de conveniência para obter um gerenciador de embedding.
    
    Args:
        config: Objeto de configuração RAGConfig.
        
    Returns:
        EmbeddingModelManager: Instância do gerenciador de embedding.
    """
    return initialize_embedding_model(config)


if __name__ == "__main__":
    # Teste do módulo
    print("🧪 Testando módulo de embedding...")
    
    from .config_manager import RAGConfig
    test_config = RAGConfig()

    try:
        manager = initialize_embedding_model(test_config)
        
        # Teste de consulta
        query = "Como implementar autenticação no FastAPI?"
        embedding = manager.embed_query(query)
        
        if embedding is not None:
            print(f"[OK] Embedding gerado com sucesso! Dimensão: {len(embedding)}")
            print(f"[INFO] Info do cache: {manager.get_cache_info()}")
        else:
            print("[ERRO] Falha ao gerar embedding.")

        # Teste de documentos
        documents = [
            "Este é o primeiro documento.",
            "Este é o segundo documento para teste."
        ]
        embeddings = manager.embed_documents(documents)
        if embeddings:
            print(f"[OK] Embeddings gerados para {len(embeddings)} documentos.")
            print(f"[INFO] Dimensão do primeiro embedding: {len(embeddings[0])}")
        else:
            print("[ERRO] Falha ao gerar embeddings para documentos.")

        manager.unload_model()
        print("Modelo descarregado para teste.")

    except Exception as e:
        print(f"[ERRO] Ocorreu um erro durante o teste: {e}")