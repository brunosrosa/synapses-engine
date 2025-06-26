# -*- coding: utf-8 -*-
"""
Gerenciador de Configuração Unificado para RAG Recoloca.ai

Este módulo implementa um sistema de configuração unificado usando Pydantic BaseSettings
com validação automática de GPU/PyTorch e otimizações específicas para RTX 2060m.

Autor: @AgenteM_DevFastAPI
Versão: 2.0 (Refatoração)
Data: Janeiro 2025
"""

import os
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass

from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DETECÇÃO E VALIDAÇÃO DE GPU/PYTORCH
# =============================================================================

@dataclass
class GPUInfo:
    """Informações sobre a GPU detectada."""
    name: str
    memory_gb: float
    compute_capability: Optional[str]
    is_rtx2060m: bool
    cuda_available: bool
    pytorch_available: bool

class GPUValidation:
    """Classe para validação e configuração de GPU/PyTorch."""
    
    @staticmethod
    def validate_pytorch_cuda() -> Dict[str, Any]:
        """Valida se PyTorch detecta CUDA corretamente."""
        validation_result = {
            "pytorch_available": False,
            "cuda_available": False,
            "cuda_version": None,
            "pytorch_version": None,
            "driver_version": None,
            "errors": []
        }
        
        try:
            import torch
            validation_result["pytorch_available"] = True
            validation_result["pytorch_version"] = torch.__version__
            
            if torch.cuda.is_available():
                validation_result["cuda_available"] = True
                validation_result["cuda_version"] = torch.version.cuda
                
                # Verificar driver NVIDIA
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    validation_result["driver_version"] = driver_version
                except ImportError:
                    validation_result["errors"].append("pynvml não disponível para verificar driver NVIDIA")
                except Exception as e:
                    validation_result["errors"].append(f"Erro ao verificar driver NVIDIA: {e}")
            else:
                validation_result["errors"].append("CUDA não disponível no PyTorch")
                
        except ImportError:
            validation_result["errors"].append("PyTorch não instalado")
        except Exception as e:
            validation_result["errors"].append(f"Erro na validação PyTorch: {e}")
            
        return validation_result
    
    @staticmethod
    def validate_gpu_memory() -> Dict[str, Any]:
        """Valida memória GPU disponível."""
        memory_info = {
            "total_memory_gb": 0.0,
            "available_memory_gb": 0.0,
            "gpu_name": "Unknown",
            "is_rtx2060m": False,
            "memory_sufficient": False,
            "errors": []
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                memory_info["gpu_name"] = device_props.name
                memory_info["total_memory_gb"] = device_props.total_memory / (1024**3)
                
                # Verificar se é RTX 2060m
                gpu_name_lower = device_props.name.lower()
                memory_info["is_rtx2060m"] = "rtx 2060" in gpu_name_lower and "mobile" in gpu_name_lower
                
                # Verificar memória disponível atual
                torch.cuda.empty_cache()
                memory_info["available_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # Considerar suficiente se tiver pelo menos 4GB
                memory_info["memory_sufficient"] = memory_info["total_memory_gb"] >= 4.0
                
            else:
                memory_info["errors"].append("CUDA não disponível")
                
        except Exception as e:
            memory_info["errors"].append(f"Erro na validação de memória GPU: {e}")
            
        return memory_info
    
    @staticmethod
    def configure_rtx2060m_optimizations() -> Dict[str, Any]:
        """Configurações específicas para RTX 2060m."""
        return {
            "device": "cuda",
            "memory_fraction": 0.8,  # Usar 80% da VRAM
            "batch_size": 16,  # Batch size otimizado para 6GB VRAM
            "precision": "fp16",  # Mixed precision para economizar memória
            "memory_efficient_attention": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 2,
            "pin_memory": True,
            "non_blocking": True,
            "torch_compile": False,  # Pode causar problemas em GPUs mais antigas
            "cache_size_mb": 512,  # Cache menor para economizar VRAM
            "enable_flash_attention": False  # Não suportado em RTX 2060m
        }
    
    @classmethod
    def get_gpu_info(cls) -> GPUInfo:
        """Obtém informações completas sobre a GPU."""
        pytorch_validation = cls.validate_pytorch_cuda()
        memory_validation = cls.validate_gpu_memory()
        
        return GPUInfo(
            name=memory_validation.get("gpu_name", "Unknown"),
            memory_gb=memory_validation.get("total_memory_gb", 0.0),
            compute_capability=None,  # TODO: Implementar detecção
            is_rtx2060m=memory_validation.get("is_rtx2060m", False),
            cuda_available=pytorch_validation.get("cuda_available", False),
            pytorch_available=pytorch_validation.get("pytorch_available", False)
        )

# =============================================================================
# CONFIGURAÇÕES PYDANTIC
# =============================================================================

class LoggingConfig(BaseSettings):
    """Configurações de logging."""
    model_config = SettingsConfigDict(env_prefix='RAG_LOG_', extra='ignore')
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file_path: Optional[Path] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    use_colors: bool = True
    enable_file_logging: bool = True
    enable_console_logging: bool = True

class EmbeddingModelConfig(BaseSettings):
    """Configurações do modelo de embedding."""
    model_config = SettingsConfigDict(env_prefix='RAG_EMBEDDING_', extra='ignore')
    
    name: str = "BAAI/bge-m3"
    max_tokens: int = 8192
    device: str = "auto"  # auto, cpu, cuda
    trust_remote_code: bool = True
    cache_dir: Optional[Path] = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    
    # Configurações específicas para GPU
    use_fp16: bool = False
    memory_efficient: bool = True

class FAISSConfig(BaseSettings):
    """Configurações do FAISS."""
    model_config = SettingsConfigDict(env_prefix='RAG_FAISS_', extra='ignore')
    
    index_type: str = "IndexFlatIP"
    metric_type: str = "METRIC_INNER_PRODUCT"
    nprobe: int = 32
    nlist: int = 100
    factory_string: str = "Flat"
    train_size: int = 10000
    ef_search: int = 64
    use_gpu: bool = False  # Será configurado automaticamente

class PyTorchConfig(BaseSettings):
    """Configurações específicas do PyTorch."""
    model_config = SettingsConfigDict(env_prefix='RAG_PYTORCH_', extra='ignore')
    
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    memory_fraction: float = 0.9
    enable_amp: bool = False  # Automatic Mixed Precision
    compile_model: bool = False
    
    # Configurações específicas para RTX 2060m
    rtx2060m_optimizations: bool = True

class RAGConfig(BaseSettings):
    """Configuração principal unificada do sistema RAG."""
    model_config = SettingsConfigDict(
        env_prefix='RAG_',
        extra='ignore',
        env_file='.env',
        env_file_encoding='utf-8'
    )
    
    # Sub-configurações
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    pytorch: PyTorchConfig = Field(default_factory=PyTorchConfig)
    
    # Diretórios (calculados automaticamente)
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[4])
    rag_infra_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    
    # Configurações gerais
    force_cpu: bool = False
    force_pytorch: bool = False
    use_optimizations: bool = True
    top_k_default: int = 5
    similarity_threshold: float = 0.2
    cache_max_size: int = 100  # MB
    
    # Configurações de chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # Configurações automáticas (computed fields)
    @computed_field
    @property
    def data_index_dir(self) -> Path:
        return self.rag_infra_root / "data" / "indexes"
    
    @computed_field
    @property
    def source_documents_dir(self) -> Path:
        return self.rag_infra_root / "knowledge_base_for_rag"
    
    @computed_field
    @property
    def logs_dir(self) -> Path:
        return self.rag_infra_root / "logs" / "application"
    
    @computed_field
    @property
    def reports_dir(self) -> Path:
        return self.rag_infra_root / "reports"
    
    @computed_field
    @property
    def pytorch_index_dir(self) -> Path:
        return self.data_index_dir / "pytorch_index_bge_m3"
    
    # Validação e configuração automática
    def model_post_init(self, __context) -> None:
        """Configuração automática pós-inicialização."""
        self._setup_directories()
        self._configure_gpu_settings()
        self._validate_environment()
    
    def _setup_directories(self) -> None:
        """Cria diretórios necessários."""
        directories = [
            self.data_index_dir,
            self.source_documents_dir,
            self.logs_dir,
            self.reports_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Diretório criado/verificado: {directory}")
    
    def _configure_gpu_settings(self) -> None:
        """Configura automaticamente as configurações de GPU."""
        gpu_info = GPUValidation.get_gpu_info()
        
        if not self.force_cpu and gpu_info.cuda_available:
            # Configurar para uso de GPU
            if gpu_info.is_rtx2060m:
                logger.info("RTX 2060m detectada - aplicando otimizações específicas")
                rtx_config = GPUValidation.configure_rtx2060m_optimizations()
                
                # Aplicar configurações RTX 2060m
                self.pytorch.batch_size = rtx_config["batch_size"]
                self.pytorch.memory_fraction = rtx_config["memory_fraction"]
                self.pytorch.enable_amp = rtx_config["precision"] == "fp16"
                self.embedding.use_fp16 = True
                self.embedding.batch_size = rtx_config["batch_size"]
                
                # FAISS-GPU pode ter problemas com RTX 2060m
                self.faiss.use_gpu = False
                self.force_pytorch = True
                
            else:
                # GPU genérica
                self.faiss.use_gpu = True
                
            self.embedding.device = "cuda"
        else:
            # Fallback para CPU
            logger.info("Configurando para uso de CPU")
            self.embedding.device = "cpu"
            self.faiss.use_gpu = False
            self.pytorch.enable_amp = False
    
    def _validate_environment(self) -> None:
        """Valida o ambiente de execução."""
        validation_results = {
            "pytorch": GPUValidation.validate_pytorch_cuda(),
            "gpu_memory": GPUValidation.validate_gpu_memory()
        }
        
        # Log dos resultados de validação
        for component, result in validation_results.items():
            if result.get("errors"):
                for error in result["errors"]:
                    logger.warning(f"Validação {component}: {error}")
            else:
                logger.info(f"Validação {component}: OK")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Retorna um resumo do status da configuração."""
        gpu_info = GPUValidation.get_gpu_info()
        
        return {
            "gpu_info": {
                "name": gpu_info.name,
                "memory_gb": gpu_info.memory_gb,
                "is_rtx2060m": gpu_info.is_rtx2060m,
                "cuda_available": gpu_info.cuda_available
            },
            "backend": "pytorch" if self.force_pytorch else "faiss",
            "device": self.embedding.device,
            "optimizations_enabled": self.use_optimizations,
            "directories": {
                "data_index": str(self.data_index_dir),
                "source_documents": str(self.source_documents_dir),
                "logs": str(self.logs_dir)
            }
        }

# =============================================================================
# FACTORY E UTILITÁRIOS
# =============================================================================

class ConfigManager:
    """Gerenciador principal de configuração."""
    
    _instance: Optional[RAGConfig] = None
    
    @classmethod
    def get_config(cls, reload: bool = False) -> RAGConfig:
        """Obtém a configuração singleton."""
        if cls._instance is None or reload:
            cls._instance = RAGConfig()
            logger.info("Configuração RAG inicializada")
        return cls._instance
    
    @classmethod
    def validate_environment(cls) -> Dict[str, Any]:
        """Executa validação completa do ambiente."""
        results = {
            "pytorch_cuda": GPUValidation.validate_pytorch_cuda(),
            "gpu_memory": GPUValidation.validate_gpu_memory(),
            "rtx2060m_config": GPUValidation.configure_rtx2060m_optimizations()
        }
        
        # Determinar status geral
        overall_status = "OK"
        for key, result in results.items():
            if isinstance(result, dict) and not result.get("success", True):
                overall_status = "WARNING"
                break
        
        results["overall_status"] = overall_status
        return results
    
    @classmethod
    def create_health_check(cls) -> Dict[str, Any]:
        """Cria um health check completo do sistema."""
        config = cls.get_config()
        validation = cls.validate_environment()
        
        return {
            "timestamp": str(Path().cwd()),
            "config_status": config.get_status_summary(),
            "validation_results": validation,
            "health": "OK" if not any(
                result.get("errors", []) for result in validation.values()
            ) else "WARNING"
        }

# Instância global para compatibilidade
rag_config = ConfigManager.get_config()

# Exports
__all__ = [
    "RAGConfig",
    "ConfigManager",
    "GPUValidation",
    "GPUInfo",
    "LoggingConfig",
    "EmbeddingModelConfig",
    "FAISSConfig",
    "PyTorchConfig",
    "rag_config"
]