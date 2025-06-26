# -*- coding: utf-8 -*-
"""
Retriever RAG para o Sistema Recoloca.ai

Este módulo é responsável por realizar consultas no índice FAISS,
retornando os documentos mais relevantes para uma query específica.

Autor: @AgenteM_DevFastAPI
Versão: 1.0
Data: Junho 2025
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int

import faiss
import numpy as np

from .config_manager import RAGConfig
from .embedding_model import EmbeddingModelManager, initialize_embedding_model
from .pytorch_gpu_retriever import PyTorchGPURetriever
from .pytorch_optimizations import OptimizedPyTorchRetriever

# Importar sistema de métricas
try:
    from .rag_metrics_integration import track_query, track_batch_query, metrics_integration
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    # Decoradores vazios se métricas não disponíveis
    def track_query(func):
        return func
    def track_batch_query(func):
        return func

# Configurar logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def _detect_gpu_compatibility(config: RAGConfig) -> bool:
    """
    Detecta se a GPU é compatível com FAISS-GPU.
    
    Returns:
        bool: True se compatível com FAISS-GPU, False caso contrário
    """
    if config.force_cpu:
        logger.info(config.status_messages["force_cpu_enabled"])
        return False

    try:
        import torch
        if not torch.cuda.is_available():
            logger.info(config.status_messages["cuda_unavailable"])
            return False
            
        # Verificar se é RTX 2060 ou similar (problemas conhecidos com FAISS-GPU)
        gpu_name = torch.cuda.get_device_name(0).lower()
        incompatible_gpus = ['rtx 2060', 'gtx 1660', 'gtx 1650']
        
        for incompatible in incompatible_gpus:
            if incompatible in gpu_name:
                logger.info(f"GPU {gpu_name} detectada - usando PyTorch em vez de FAISS-GPU")
                return False
                
        # Tentar importar FAISS-GPU
        try:
            import faiss
            if hasattr(faiss, 'StandardGpuResources'):
                logger.info(f"GPU {gpu_name} compatível com FAISS-GPU")
                return True
        except Exception as e:
            logger.warning(f"FAISS-GPU não disponível: {e}")
            
        return False
        
    except Exception as e:
        logger.warning(f"Erro na detecção de GPU: {e}")
        return False

class RAGRetriever:
    """
    Classe principal para recuperação de documentos usando FAISS ou PyTorch.
    Implementa fallback inteligente baseado na compatibilidade da GPU.
    """

    def __init__(self, config: RAGConfig):
        """
        Inicializa o retriever com base nas configurações fornecidas.
        
        Args:
            config: Instância de RAGConfig contendo as configurações do sistema.
        """
        self.config = config
        self.embedding_manager: Optional[EmbeddingModelManager] = None
        self.pytorch_retriever: Optional[PyTorchGPURetriever] = None
        self.index = None
        
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.index_metadata: Dict[str, Any] = {}
        self.is_loaded = False
        
        self._query_cache: Dict[str, List[SearchResult]] = {}
        self._cache_max_size = self.config.cache_max_size
        
        self.use_pytorch = self._determine_backend_from_config()
        from .constants import STATUS_MESSAGES
        self.STATUS_MESSAGES = STATUS_MESSAGES
        
        logger.info(f"Backend selecionado: {'PyTorch' if self.use_pytorch else 'FAISS'}")
        
    @property
    def is_initialized(self) -> bool:
        """
        Verifica se o retriever foi inicializado corretamente.
        """
        if self.use_pytorch:
            return self.pytorch_retriever is not None and hasattr(self.pytorch_retriever, 'embedding_manager') and self.pytorch_retriever.embedding_manager is not None
        else:
            return self.embedding_manager is not None
        


    def _determine_backend_from_config(self) -> bool:
        """
        Determina se deve usar PyTorch com base na configuração.
        """
        if self.config.force_cpu:
            logger.info("Configuração: Forçando uso de CPU.")
            return True
        if self.config.force_pytorch:
            logger.info("Configuração: Forçando uso de PyTorch.")
            return True
        
        # Se GPU estiver habilitada na config, tentar detectar compatibilidade FAISS-GPU
        if not self.config.force_cpu:
            if _detect_gpu_compatibility(self.config):
                logger.info("Configuração: GPU habilitada e compatível com FAISS-GPU. Usando FAISS.")
                return False # Usar FAISS
            else:
                logger.info("Configuração: GPU habilitada, mas FAISS-GPU incompatível. Usando PyTorch.")
                return True # Usar PyTorch
        
        logger.info("Configuração: GPU desabilitada. Usando PyTorch (CPU). Ou PyTorch é o backend padrão.")
        return True # Default para PyTorch (CPU) se GPU desabilitada ou PyTorch é o backend padrão

    def _initialize_backend(self):
        """
        Inicializa o backend apropriado (FAISS ou PyTorch) com base na configuração.
        """
        if self.use_pytorch:
            if self.config.use_optimizations:
                logger.info("Usando OptimizedPyTorchRetriever")
                self.pytorch_retriever = OptimizedPyTorchRetriever(
                    force_cpu=self.config.force_cpu,
                    cache_enabled=True,
                    cache_dir=str(Path(self.config.data_index_dir) / "cache"),
                    cache_ttl=3600,
                    batch_size=5,
                    max_workers=10
                )
            else:
                logger.info("Usando PyTorchGPURetriever")
                self.pytorch_retriever = PyTorchGPURetriever(
                    config=self.config
                )
            self.index = None
        else:
            logger.info("Usando FAISS")
            self.pytorch_retriever = None
            self.index = None
    
    def initialize(self) -> bool:
        """
        Inicializa o modelo de embedding e o backend apropriado.
        
        Returns:
            bool: True se inicializado com sucesso
        """
        try:
            logger.info("--- Iniciando RAGRetriever --- ")
            self._initialize_backend() # Mover a inicialização do backend para cá

            logger.info("Inicializando modelo de embedding...")
            
            if self.use_pytorch:
                if not self.pytorch_retriever:
                    logger.error("Pytorch retriever não foi instanciado em _initialize_backend.")
                    return False
                success = self.pytorch_retriever.initialize()
                if not success:
                    logger.error("Falha ao inicializar PyTorchGPURetriever")
                    return False
                self.embedding_manager = self.pytorch_retriever.base_retriever.embedding_manager
                logger.info("[OK] PyTorchGPURetriever e EmbeddingManager inicializados com sucesso")
            else:
                self.embedding_manager = initialize_embedding_model(self.config)
                if not self.embedding_manager or not self.embedding_manager.is_loaded:
                    logger.error("Falha ao inicializar EmbeddingModelManager para FAISS")
                    return False
                logger.info("[OK] EmbeddingModelManager para FAISS inicializado com sucesso")

            if not self.embedding_manager or not self.embedding_manager.is_loaded:
                 logger.error("Falha crítica: Embedding manager não está disponível após a inicialização.")
                 return False

            logger.info("[OK] RAGRetriever inicializado com sucesso.")
            return True
            
        except Exception as e:
            logger.error(f"Erro fatal na inicialização do RAGRetriever: {str(e)}", exc_info=True)
            return False
    
    def load_index(self) -> bool:
        """
        Carrega o índice do disco (FAISS ou PyTorch).
        
        Returns:
            bool: True se carregado com sucesso
        """
        try:
            logger.info(f"DEBUG: use_pytorch = {self.use_pytorch}")
            logger.info(f"DEBUG: pytorch_retriever = {self.pytorch_retriever}")
            
            if self.use_pytorch:
                # Carregar usando PyTorchGPURetriever
                logger.info("Carregando índice PyTorch...")
                
                if not self.pytorch_retriever:
                    logger.error("[ERROR] pytorch_retriever não está inicializado")
                    return False
                    
                success = self.pytorch_retriever.load_index()
                logger.info(f"DEBUG: pytorch_retriever.load_index() retornou: {success}")
                
                if success:
                    # Sincronizar dados com o retriever principal
                    # Para OptimizedPyTorchRetriever, acessar através do base_retriever
                    if hasattr(self.pytorch_retriever, 'base_retriever'):
                        # É um OptimizedPyTorchRetriever
                        self.documents = self.pytorch_retriever.base_retriever.documents
                        self.metadata = self.pytorch_retriever.base_retriever.metadata
                        self.index_metadata = self.pytorch_retriever.base_retriever.index_metadata
                    else:
                        # É um PyTorchGPURetriever direto
                        self.documents = self.pytorch_retriever.documents
                        self.metadata = self.pytorch_retriever.metadata
                        self.index_metadata = self.pytorch_retriever.index_metadata
                    
                    self.is_loaded = True
                    logger.info(f"[OK] Índice PyTorch carregado: {len(self.documents)} documentos")
                    return True
                else:
                    logger.error("[ERROR] Falha ao carregar índice PyTorch")
                    return False
            else:
                # Carregar usando FAISS (código original)
                logger.info("Carregando índice FAISS...")
                return self._load_faiss_index()
                
        except Exception as e:
            logger.error(f"Erro ao carregar índice: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.is_loaded = False
            return False
            
    def _load_faiss_index(self) -> bool:
        """
        Carrega o índice FAISS e os dados associados com logging detalhado.
        """
        logger.info(f"Iniciando carregamento do índice FAISS de: {self.config.faiss_index_dir}")
        start_time = time.time()
        try:
            # Construir caminhos completos
            faiss_index_path = Path(self.config.faiss_index_dir) / self.config.faiss_index_file
            documents_path = Path(self.config.faiss_index_dir) / self.config.faiss_documents_file
            metadata_path = Path(self.config.faiss_index_dir) / self.config.faiss_metadata_file
            embeddings_path = Path(self.config.faiss_index_dir) / self.config.faiss_embeddings_file
            mapping_path = Path(self.config.faiss_index_dir) / self.config.faiss_mapping_file

            # 1. Carregar índice FAISS
            if not faiss_index_path.exists():
                logger.error(f"[FAISS Load] Arquivo de índice não encontrado: {faiss_index_path}")
                return False
            self.index = faiss.read_index(str(faiss_index_path))
            logger.info(f"[FAISS Load] Índice FAISS carregado com {self.index.ntotal} vetores.")

            # 2. Carregar documentos
            if not documents_path.exists():
                logger.error(f"[FAISS Load] Arquivo de documentos não encontrado: {documents_path}")
                return False
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            logger.info(f"[FAISS Load] Documentos carregados: {len(self.documents)} chunks.")

            # 3. Carregar metadados dos documentos
            if not metadata_path.exists():
                logger.error(f"[FAISS Load] Arquivo de metadados não encontrado: {metadata_path}")
                return False
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"[FAISS Load] Metadados carregados: {len(self.metadata)} entradas.")

            # 4. Carregar embeddings (necessário para FAISS)
            if not embeddings_path.exists():
                logger.error(f"[FAISS Load] Arquivo de embeddings não encontrado: {embeddings_path}")
                return False
            self.embeddings = np.load(embeddings_path)
            logger.info(f"[FAISS Load] Embeddings carregados: {self.embeddings.shape[0]} vetores.")

            # 5. Carregar mapeamento (necessário para FAISS)
            if not mapping_path.exists():
                logger.error(f"[FAISS Load] Arquivo de mapeamento não encontrado: {mapping_path}")
                return False
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            logger.info(f"[FAISS Load] Mapeamento carregado: {len(self.mapping)} entradas.")

            # 6. Carregar metadados do índice (se existirem)
            index_metadata_path = Path(self.config.faiss_index_dir) / "index_metadata.json" # Nome fixo por enquanto
            if not index_metadata_path.exists():
                logger.warning(f"[FAISS Load] Arquivo de metadados do índice não encontrado: {index_metadata_path}")
                self.index_metadata = {} # Definir como vazio se não existir
            else:
                with open(index_metadata_path, 'r', encoding='utf-8') as f:
                    self.index_metadata = json.load(f)
                logger.info(f"[FAISS Load] Metadados do índice carregados.")

            # 7. Verificar consistência
            if len(self.documents) != len(self.metadata):
                logger.warning(f"[FAISS Consistency] Inconsistência: {len(self.documents)} documentos vs {len(self.metadata)} metadados.")
            if self.index.ntotal != len(self.documents):
                logger.warning(f"[FAISS Consistency] Inconsistência: {self.index.ntotal} vetores vs {len(self.documents)} documentos.")

            elapsed_time = time.time() - start_time
            logger.info(f"[FAISS Load] Índice FAISS e dados carregados com sucesso em {elapsed_time:.2f}s.")
            self.is_loaded = True
            return True

        except FileNotFoundError as e:
            logger.error(f"[FAISS Load Error] Arquivo não encontrado durante o carregamento: {e}", exc_info=True)
            self.is_loaded = False
            return False
        except json.JSONDecodeError as e:
            logger.error(f"[FAISS Load Error] Erro de decodificação JSON, o arquivo pode estar corrompido: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"[FAISS Load Error] Erro inesperado ao carregar índice FAISS: {e}", exc_info=True)
            return False
    
    def search(self, query: str, top_k: Optional[int] = None, 
                min_score: Optional[float] = None, 
                category_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Realiza busca semântica no índice.
        
        Args:
            query: Consulta de busca
            top_k: Número de resultados a retornar (usa config.rag_default_top_k se None)
            min_score: Score mínimo de similaridade (usa config.rag_min_similarity_score se None)
            category_filter: Filtro por categoria (opcional)
            
        Returns:
            List[SearchResult]: Lista de resultados ordenados por relevância
        """
        if not self.is_loaded:
            logger.error("Índice não carregado. Chame load_index() primeiro.")
            return []
        
        # Usar valores da configuração se não forem fornecidos
        current_top_k = top_k if top_k is not None else self.config.top_k_default
        current_min_score = min_score if min_score is not None else self.config.similarity_threshold

        # Validar parâmetros
        current_top_k = min(max(1, current_top_k), 50)  # Valor máximo padrão
        current_min_score = max(0.0, min(1.0, current_min_score))
        
        # Verificar cache
        cache_key = f"{query}_{top_k}_{min_score}_{category_filter}"
        cache_hit = False
        if cache_key in self._query_cache:
            cache_hit = True
            logger.debug(f"Resultado encontrado no cache para: {query[:50]}...")
            cached_results = self._query_cache[cache_key]
            # Adicionar informação de cache hit aos metadados
            for result in cached_results:
                if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                    result.metadata['cache_hit'] = True
            return cached_results
        
        try:
            logger.info(self.STATUS_MESSAGES["query_start"].format(query=query[:100]))
            start_time = time.time()
            
            if self.use_pytorch:
                # Usar PyTorchGPURetriever
                pytorch_results = self.pytorch_retriever.search(
                    query=query,
                    top_k=current_top_k,
                    min_score=current_min_score
                )
                
                # Converter para SearchResult e aplicar filtro de categoria
                results = []
                for pytorch_result in pytorch_results:
                    # Aplicar filtro de categoria se especificado
                    if category_filter and pytorch_result.metadata.get("category") != category_filter:
                        continue
                        
                    # Converter para o formato SearchResult do RAGRetriever
                    result = SearchResult(
                        content=pytorch_result.content,
                        metadata=pytorch_result.metadata,
                        score=pytorch_result.score,
                        rank=pytorch_result.rank
                    )
                    results.append(result)
                    
                    if len(results) >= top_k:
                        break
                        
            else:
                # Usar FAISS (código original)
                results = self._search_faiss(query, top_k, min_score, category_filter)
            
            elapsed_time = time.time() - start_time
            logger.info(self.STATUS_MESSAGES["query_complete"].format(count=len(results)))
            logger.info(f"Busca realizada em {elapsed_time:.3f}s")
            
            # Adicionar informação de cache miss aos metadados
            for result in results:
                if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                    result.metadata['cache_hit'] = False
            
            # Cache do resultado
            self._cache_result(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(self.STATUS_MESSAGES["query_error"].format(error=str(e)))
            return []
    
    def batch_search(self, queries: List[str], top_k: Optional[int] = None,
                     min_score: Optional[float] = None,
                     category_filter: Optional[str] = None) -> List[List[SearchResult]]:
        """
        Realiza busca semântica para múltiplas queries em lote.
        
        Args:
            queries: Lista de consultas de busca
            top_k: Número de resultados a retornar por query (usa config.rag_default_top_k se None)
            min_score: Score mínimo de similaridade (usa config.rag_min_similarity_score se None)
            category_filter: Filtro por categoria (opcional)
            
        Returns:
            List[List[SearchResult]]: Lista de listas de resultados, uma para cada query
        """
        if not self.is_loaded:
            logger.error("Índice não carregado. Chame load_index() primeiro.")
            return [[] for _ in queries]

        # Usar valores da configuração se não forem fornecidos
        current_top_k = top_k if top_k is not None else self.config.top_k_default
        current_min_score = min_score if min_score is not None else self.config.similarity_threshold

        # Validar parâmetros
        current_top_k = min(max(1, current_top_k), 50)  # Valor máximo padrão
        current_min_score = max(0.0, min(1.0, current_min_score))

        results_by_query: List[List[SearchResult]] = []
        for query in queries:
            results_by_query.append(self.search(query, current_top_k, current_min_score, category_filter))
        return results_by_query
            
    def _search_faiss(self, query: str, current_top_k: int, current_min_score: float, 
                       category_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Realiza a busca no índice FAISS.
        """
        if not self.index or not self.embedding_manager:
            logger.error("Índice FAISS ou EmbeddingManager não inicializado.")
            return []

        query_embedding = self.embedding_manager.embed_query(query)
        if query_embedding is None:
            logger.error("Não foi possível gerar embedding para a query.")
            return []

        query_embedding = np.array([query_embedding]).astype('float32')

        # Realizar busca FAISS
        distances, indices = self.index.search(query_embedding, current_top_k)

        results: List[SearchResult] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS retorna -1 para resultados inválidos
                continue

            score = 1 - distances[0][i]  # Converter distância para score de similaridade (0 a 1)
            if score < current_min_score:
                continue

            # Mapear o índice FAISS de volta para o índice original do documento
            original_doc_idx = self.mapping.get(str(idx))
            if original_doc_idx is None:
                logger.warning(f"Índice FAISS {idx} não encontrado no mapeamento.")
                continue

            document = self.documents[original_doc_idx]
            metadata = self.metadata[original_doc_idx]

            # Aplicar filtro de categoria
            if category_filter and metadata.get("category") != category_filter:
                continue

            results.append(SearchResult(
                content=document,
                metadata=metadata,
                score=score,
                rank=len(results) + 1
            ))
        
        # Ordenar resultados por score (decrescente)
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def search_by_document(self, document_pattern: str, top_k: int = 5) -> List[SearchResult]:
        """
        Busca documentos por padrão no nome/caminho.
        
        Args:
            document_pattern: Padrão para buscar no nome do documento
            top_k: Número máximo de resultados
            
        Returns:
            List[SearchResult]: Lista de resultados encontrados
        """
        if not self.is_loaded:
            logger.error("Índice não carregado.")
            return []
        
        try:
            results = []
            pattern_lower = document_pattern.lower()
            
            for idx, metadata in enumerate(self.metadata):
                source = metadata.get("source", "").lower()
                title = metadata.get("title", "").lower()
                
                if pattern_lower in source or pattern_lower in title:
                    if idx < len(self.documents):
                        result = SearchResult(
                            content=self.documents[idx],
                            metadata=metadata,
                            score=1.0,  # Score fixo para busca por documento
                            rank=len(results) + 1
                        )
                        results.append(result)
                        
                        if len(results) >= top_k:
                            break
            
            logger.info(f"Encontrados {len(results)} documentos para padrão: {document_pattern}")
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca por documento: {str(e)}")
            return []
    
    def get_document_list(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retorna lista de documentos indexados.
        
        Args:
            category: Filtro por categoria (opcional)
            
        Returns:
            List[Dict]: Lista de informações dos documentos
        """
        if not self.is_loaded:
            logger.error("Índice não carregado.")
            return []
        
        try:
            documents_info = []
            
            # Agrupar chunks por documento fonte
            doc_groups = {}
            for metadata in self.metadata:
                source = metadata.get("source", "unknown")
                doc_category = metadata.get("category", "general")
                
                # Aplicar filtro de categoria se especificado
                if category and doc_category != category:
                    continue
                
                if source not in doc_groups:
                    doc_groups[source] = {
                        "source": source,
                        "title": metadata.get("title", "Sem título"),
                        "category": doc_category,
                        "file_type": metadata.get("file_type", "unknown"),
                        "chunks_count": 0,
                        "last_updated": metadata.get("timestamp", "unknown")
                    }
                
                doc_groups[source]["chunks_count"] += 1
            
            documents_info = list(doc_groups.values())
            documents_info.sort(key=lambda x: x["source"])
            
            logger.info(f"Retornando informações de {len(documents_info)} documentos")
            return documents_info
            
        except Exception as e:
            logger.error(f"Erro ao obter lista de documentos: {str(e)}")
            return []
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o índice carregado.
        
        Returns:
            Dict: Informações do índice
        """
        if not self.is_loaded:
            return {
                "loaded": False, 
                "error": "Índice não carregado",
                "backend": None,
                "device": None
            }
        
        base_info = {
            "loaded": True,
            "backend": "PyTorch_Optimized" if (self.use_pytorch and self.config.use_optimizations) else ("PyTorch" if self.use_pytorch else "FAISS"),
            "total_documents": len(self.documents),
            "total_metadata": len(self.metadata),
            "embedding_model": self.index_metadata.get("embedding_model", "unknown"),
            "created_at": self.index_metadata.get("created_at", "unknown"),
            "index_dimension": self.index_metadata.get("index_dimension", 0),
            "chunk_size": self.index_metadata.get("chunk_size", 0),
            "chunk_overlap": self.index_metadata.get("chunk_overlap", 0),
            "cache_size": len(self._query_cache),
            "optimizations_enabled": self.config.use_optimizations
        }
        
        if self.use_pytorch:
            # Informações específicas do PyTorch
            try:
                pytorch_info = self.pytorch_retriever.get_index_info()
                
                # Garantir que sempre temos os campos essenciais
                device = pytorch_info.get("device", "unknown")
                if device == "unknown" and hasattr(self.pytorch_retriever, 'base_retriever'):
                    # Para OptimizedPyTorchRetriever, tentar obter do base_retriever
                    base_pytorch_info = self.pytorch_retriever.base_retriever.get_index_info()
                    device = base_pytorch_info.get("device", "unknown")
                
                base_info.update({
                    "total_vectors": pytorch_info.get("total_vectors", pytorch_info.get("documents_count", len(self.documents))),
                    "device": device,
                    "gpu_available": pytorch_info.get("gpu_available", False)
                })
            except Exception as e:
                logger.warning(f"Erro ao obter informações do PyTorch retriever: {e}")
                base_info.update({
                    "total_vectors": len(self.documents),
                    "device": "unknown",
                    "gpu_available": False
                })
        else:
            # Informações específicas do FAISS
            base_info.update({
                "total_vectors": self.index.ntotal if self.index else 0,
                "index_type": self.index.get_type() if self.index else "unknown",
                "device": "CPU",
                "gpu_available": False # FAISS-GPU é tratado como um backend separado
            })
            
        return base_info

    def _cache_result(self, key: str, results: List[SearchResult]):
        """
        Adiciona um resultado de busca ao cache.
        """
        if len(self._query_cache) >= self._cache_max_size:
            # Remover o item mais antigo (FIFO)
            oldest_key = next(iter(self._query_cache))
            self._query_cache.pop(oldest_key)
        self._query_cache[key] = results


class SearchResult:
    """
    Classe para representar um resultado de busca.
    """

    RESULT_TEMPLATE = """
--- Documento: {source} - Seção: {section} (Score: {score:.2f})
{content}
"""

    def __init__(self, content: str, metadata: Dict[str, Any], score: float, rank: int):
        self.content = content
        self.metadata = metadata
        self.score = score
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o resultado para dicionário.
        
        Returns:
            Dict: Representação em dicionário do resultado
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "rank": self.rank
        }


# =============================================================================
# FASE 2: RAGRetrieverFactory e Funções Globais Temporárias
# =============================================================================

class RAGRetrieverFactory:
    """
    Factory para criação e gerenciamento de instâncias RAGRetriever.
    
    Implementa padrão Singleton com injeção de dependências para garantir
    uma única instância do retriever por configuração, com inicialização
    assíncrona e gerenciamento robusto de recursos.
    
    Características:
    - Singleton por configuração
    - Inicialização assíncrona
    - Injeção de dependências
    - Tratamento robusto de erros
    - Cleanup automático de recursos
    """
    
    _instances: Dict[str, 'RAGRetrieverFactory'] = {}
    _lock = asyncio.Lock()
    
    def __init__(self, config: RAGConfig, embedding_manager: Optional[EmbeddingModelManager] = None):
        """
        Inicializa a factory com configuração e dependências.
        
        Args:
            config: Configuração RAG
            embedding_manager: Gerenciador de embeddings (opcional)
        """
        self.config = config
        self.embedding_manager = embedding_manager
        self._retriever: Optional[RAGRetriever] = None
        self._initialized = False
        self._initialization_error: Optional[Exception] = None
    
    @classmethod
    async def get_instance(cls, config: RAGConfig, 
                          embedding_manager: Optional[EmbeddingModelManager] = None) -> 'RAGRetrieverFactory':
        """
        Obtém instância singleton da factory baseada na configuração.
        
        Args:
            config: Configuração RAG
            embedding_manager: Gerenciador de embeddings (opcional)
            
        Returns:
            Instância da factory
        """
        # Criar chave baseada no diretório de índices e configuração de backend
        backend_type = "pytorch" if config.force_pytorch else "faiss"
        config_key = f"{config.data_index_dir}_{backend_type}"
        
        async with cls._lock:
            if config_key not in cls._instances:
                cls._instances[config_key] = cls(config, embedding_manager)
            return cls._instances[config_key]
    
    async def get_retriever(self) -> RAGRetriever:
        """
        Obtém instância do retriever, inicializando se necessário.
        
        Returns:
            Instância configurada do RAGRetriever
            
        Raises:
            RuntimeError: Se a inicialização falhar
        """
        if not self._initialized:
            await self._initialize_retriever()
        
        if self._initialization_error:
            raise RuntimeError(f"Falha na inicialização do retriever: {self._initialization_error}")
        
        if self._retriever is None:
            raise RuntimeError("Retriever não foi inicializado corretamente")
        
        return self._retriever
    
    async def _initialize_retriever(self) -> None:
        """
        Inicializa o retriever de forma assíncrona.
        """
        try:
            # Inicializar embedding manager se não fornecido
            if self.embedding_manager is None:
                self.embedding_manager = initialize_embedding_model(self.config)
            
            # Criar instância do retriever
            self._retriever = RAGRetriever(self.config)
            
            # Inicializar o retriever
            init_success = await asyncio.get_event_loop().run_in_executor(
                None, self._retriever.initialize
            )
            
            if not init_success:
                raise RuntimeError("Falha na inicialização do retriever")
            
            # Carregar o índice
            load_success = await asyncio.get_event_loop().run_in_executor(
                None, self._retriever.load_index
            )
            
            if not load_success:
                logger.warning("Falha ao carregar índice existente - será necessário indexar documentos")
            
            self._initialized = True
            logger.info("RAGRetriever inicializado com sucesso via factory")
            
        except Exception as e:
            self._initialization_error = e
            logger.error(f"Erro na inicialização do RAGRetriever: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """
        Limpa recursos da factory.
        """
        if self._retriever:
            # Cleanup do retriever se tiver método cleanup
            if hasattr(self._retriever, 'cleanup'):
                await asyncio.get_event_loop().run_in_executor(
                    None, self._retriever.cleanup
                )
        
        self._retriever = None
        self._initialized = False
        self._initialization_error = None
    
    @classmethod
    async def cleanup_all(cls) -> None:
        """
        Limpa todas as instâncias da factory.
        """
        async with cls._lock:
            for factory in cls._instances.values():
                await factory.cleanup()
            cls._instances.clear()
    
    def is_initialized(self) -> bool:
        """
        Verifica se o factory foi inicializado.
        
        Returns:
            bool: True se inicializado
        """
        return self._initialized and self._retriever is not None
    
    def reset(self):
        """
        Reseta o factory, forçando recriação na próxima chamada.
        """
        logger.info("🔄 Resetando RAGRetrieverFactory...")
        self._retriever = None
        self._initialized = False


# =============================================================================
# FUNÇÕES GLOBAIS TEMPORÁRIAS (DEPRECATED)
# TODO: Remover após migração completa para RAGRetrieverFactory
# =============================================================================

# Instância global da factory (temporária)
_factory_instance: Optional[RAGRetrieverFactory] = None
_global_config: Optional[RAGConfig] = None

async def initialize_retriever(config: Optional[RAGConfig] = None, 
                              embedding_manager: Optional[EmbeddingModelManager] = None) -> bool:
    """
    DEPRECATED: Função global para inicializar o retriever.
    
    TODO: Migrar para uso direto do RAGRetrieverFactory.
    
    Args:
        config: Configuração RAG (opcional)
        embedding_manager: Gerenciador de embeddings (opcional)
        
    Returns:
        bool: True se inicializado com sucesso
    """
    global _factory_instance, _global_config
    
    try:
        if config is None:
            from .config_manager import RAGConfig
            config = RAGConfig()
        
        _global_config = config
        _factory_instance = await RAGRetrieverFactory.get_instance(config, embedding_manager)
        
        logger.warning("⚠️ DEPRECATED: Usando função global initialize_retriever. Migre para RAGRetrieverFactory.")
        return True
        
    except Exception as e:
        logger.error(f"Erro na inicialização global do retriever: {e}")
        return False

async def get_retriever() -> Optional[RAGRetriever]:
    """
    DEPRECATED: Função global para obter instância do retriever.
    
    TODO: Migrar para uso direto do RAGRetrieverFactory.
    
    Returns:
        Optional[RAGRetriever]: Instância do retriever ou None
    """
    global _factory_instance, _global_config
    
    try:
        if _factory_instance is None:
            if _global_config is None:
                from .config_manager import RAGConfig
                _global_config = RAGConfig()
            
            _factory_instance = await RAGRetrieverFactory.get_instance(_global_config)
        
        retriever = await _factory_instance.get_retriever()
        logger.warning("⚠️ DEPRECATED: Usando função global get_retriever. Migre para RAGRetrieverFactory.")
        return retriever
        
    except Exception as e:
        logger.error(f"Erro ao obter retriever global: {e}")
        return None

async def search_documents(query: str, top_k: int = 5, min_score: float = 0.1) -> List[SearchResult]:
    """
    DEPRECATED: Função global para busca de documentos.
    
    TODO: Migrar para uso direto do RAGRetriever via factory.
    
    Args:
        query: Consulta de busca
        top_k: Número de resultados
        min_score: Score mínimo
        
    Returns:
        List[SearchResult]: Resultados da busca
    """
    try:
        retriever = await get_retriever()
        if retriever is None:
            logger.error("Retriever não disponível para busca global")
            return []
        
        results = retriever.search(query, top_k, min_score)
        logger.warning("⚠️ DEPRECATED: Usando função global search_documents. Migre para RAGRetrieverFactory.")
        return results
        
    except Exception as e:
        logger.error(f"Erro na busca global de documentos: {e}")
        return []