# -*- coding: utf-8 -*-
"""
Configuração Avançada de Otimizações RAG - Recoloca.ai

Este módulo define configurações otimizadas para o sistema RAG,
incluindo métricas de monitoramento, configurações de cache TTL,
batch processing e preparação para testes de carga.

Autor: @AgenteM_DevFastAPI
Versão: 1.0
Data: Junho 2025
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

# Configurações base do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
METRICS_DIR = PROJECT_ROOT / "metrics"

# Criar diretórios se não existirem
for directory in [CACHE_DIR, LOGS_DIR, METRICS_DIR]:
    directory.mkdir(exist_ok=True)

@dataclass
class CacheConfig:
    """Configurações do sistema de cache."""
    enabled: bool = True
    max_entries: int = 1000
    ttl_seconds: int = 3600  # 1 hora
    auto_save_interval: int = 300  # 5 minutos
    cache_dir: str = str(CACHE_DIR)
    
    # Configurações específicas para RTX 2060
    memory_limit_mb: int = 4096  # Limite de memória para cache
    cleanup_threshold: float = 0.8  # Limpar quando usar 80% da memória

@dataclass
class BatchConfig:
    """Configurações de processamento em lote."""
    enabled: bool = True
    default_batch_size: int = 5
    max_batch_size: int = 20
    max_workers: int = 4
    timeout_seconds: int = 30
    
    # Configurações adaptativas
    adaptive_sizing: bool = True
    min_batch_size: int = 2
    performance_threshold_ms: float = 500.0

@dataclass
class MonitoringConfig:
    """Configurações de monitoramento e métricas."""
    enabled: bool = True
    metrics_dir: str = str(METRICS_DIR)
    log_level: str = "INFO"
    
    # Métricas de cache
    track_cache_metrics: bool = True
    cache_metrics_interval: int = 60  # segundos
    
    # Métricas de performance
    track_performance_metrics: bool = True
    performance_metrics_interval: int = 30  # segundos
    
    # Alertas
    enable_alerts: bool = True
    cache_hit_rate_threshold: float = 0.6  # Alertar se hit rate < 60%
    avg_response_time_threshold: float = 1000.0  # ms
    
    # Exportação de métricas
    export_format: str = "json"  # json, csv, prometheus
    export_interval: int = 300  # 5 minutos

@dataclass
class GPUConfig:
    """Configurações específicas para GPU."""
    force_cpu: bool = False
    force_pytorch: bool = False
    
    # Configurações RTX 2060
    memory_fraction: float = 0.8  # Usar 80% da VRAM
    allow_growth: bool = True
    
    # Otimizações específicas
    use_mixed_precision: bool = True
    enable_cudnn_benchmark: bool = True
    
    # Fallback automático
    auto_fallback_cpu: bool = True
    memory_threshold_mb: int = 500  # Fallback se memória livre < 500MB

@dataclass
class LoadTestConfig:
    """Configurações para testes de carga."""
    enabled: bool = False
    
    # Cenários de teste
    concurrent_users: int = 10
    queries_per_user: int = 50
    ramp_up_time: int = 30  # segundos
    test_duration: int = 300  # 5 minutos
    
    # Queries de teste
    test_queries_file: str = str(PROJECT_ROOT / "tests" / "load_test_queries.json")
    
    # Métricas de teste
    target_response_time_p95: float = 1000.0  # ms
    target_throughput_qps: float = 20.0  # queries por segundo
    max_error_rate: float = 0.01  # 1%

class RAGOptimizationConfig:
    """Configuração principal do sistema de otimizações RAG."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Inicializa configuração.
        
        Args:
            config_file: Caminho para arquivo de configuração personalizada
        """
        # Configurações padrão
        self.cache = CacheConfig()
        self.batch = BatchConfig()
        self.monitoring = MonitoringConfig()
        self.gpu = GPUConfig()
        self.load_test = LoadTestConfig()
        
        # Carregar configurações personalizadas se fornecidas
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Aplicar configurações específicas do ambiente
        self._apply_environment_config()
    
    def _apply_environment_config(self):
        """Aplica configurações específicas do ambiente."""
        # Detectar se é ambiente de desenvolvimento ou produção
        is_dev = os.getenv("RAG_ENV", "development") == "development"
        
        if is_dev:
            # Configurações para desenvolvimento
            self.cache.ttl_seconds = 1800  # 30 minutos
            self.monitoring.log_level = "DEBUG"
            self.load_test.enabled = True
        else:
            # Configurações para produção
            self.cache.ttl_seconds = 7200  # 2 horas
            self.cache.max_entries = 2000
            self.monitoring.log_level = "INFO"
            self.batch.max_batch_size = 50
    
    def get_optimized_config_for_rtx2060(self) -> Dict[str, Any]:
        """Retorna configuração otimizada para RTX 2060.
        
        Returns:
            Dict com configurações otimizadas
        """
        return {
            "use_optimizations": True,
            "cache_enabled": True,
            "batch_size": 8,  # Otimizado para RTX 2060
            "force_pytorch": True,  # RTX 2060 tem problemas com FAISS-GPU
            "memory_fraction": 0.75,
            "cache_config": asdict(self.cache),
            "batch_config": asdict(self.batch),
            "monitoring_config": asdict(self.monitoring),
            "gpu_config": asdict(self.gpu)
        }
    
    def save_to_file(self, filepath: str):
        """Salva configuração em arquivo JSON.
        
        Args:
            filepath: Caminho do arquivo
        """
        config_dict = {
            "cache": asdict(self.cache),
            "batch": asdict(self.batch),
            "monitoring": asdict(self.monitoring),
            "gpu": asdict(self.gpu),
            "load_test": asdict(self.load_test)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        """Carrega configuração de arquivo JSON.
        
        Args:
            filepath: Caminho do arquivo
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        if "cache" in config_dict:
            self.cache = CacheConfig(**config_dict["cache"])
        if "batch" in config_dict:
            self.batch = BatchConfig(**config_dict["batch"])
        if "monitoring" in config_dict:
            self.monitoring = MonitoringConfig(**config_dict["monitoring"])
        if "gpu" in config_dict:
            self.gpu = GPUConfig(**config_dict["gpu"])
        if "load_test" in config_dict:
            self.load_test = LoadTestConfig(**config_dict["load_test"])
    
    def validate_config(self) -> Dict[str, Any]:
        """Valida configuração e retorna relatório.
        
        Returns:
            Dict com resultado da validação
        """
        issues = []
        warnings = []
        
        # Validar cache
        if self.cache.ttl_seconds < 300:
            warnings.append("TTL do cache muito baixo (< 5 min)")
        
        if self.cache.max_entries > 5000:
            warnings.append("Número máximo de entradas do cache muito alto")
        
        # Validar batch
        if self.batch.max_batch_size > 100:
            issues.append("Tamanho máximo de batch muito alto")
        
        # Validar GPU
        if self.gpu.memory_fraction > 0.9:
            warnings.append("Fração de memória GPU muito alta")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

# Instância global da configuração
rag_config = RAGOptimizationConfig()

# Configurações recomendadas para diferentes cenários
RECOMMENDED_CONFIGS = {
    "development": {
        "cache_ttl": 1800,  # 30 min
        "batch_size": 5,
        "log_level": "DEBUG",
        "enable_load_test": True
    },
    "production": {
        "cache_ttl": 7200,  # 2 horas
        "batch_size": 10,
        "log_level": "INFO",
        "enable_load_test": False
    },
    "rtx_2060_optimized": {
        "use_optimizations": True,
        "force_pytorch": True,
        "batch_size": 8,
        "memory_fraction": 0.75,
        "cache_ttl": 3600,  # 1 hora
        "max_cache_entries": 1500
    }
}

def get_config_for_environment(env: str = "development") -> RAGOptimizationConfig:
    """Retorna configuração otimizada para o ambiente especificado.
    
    Args:
        env: Ambiente (development, production, rtx_2060_optimized)
        
    Returns:
        Configuração otimizada
    """
    config = RAGOptimizationConfig()
    
    if env in RECOMMENDED_CONFIGS:
        recommended = RECOMMENDED_CONFIGS[env]
        
        # Aplicar configurações recomendadas
        if "cache_ttl" in recommended:
            config.cache.ttl_seconds = recommended["cache_ttl"]
        if "batch_size" in recommended:
            config.batch.default_batch_size = recommended["batch_size"]
        if "log_level" in recommended:
            config.monitoring.log_level = recommended["log_level"]
        if "enable_load_test" in recommended:
            config.load_test.enabled = recommended["enable_load_test"]
        if "memory_fraction" in recommended:
            config.gpu.memory_fraction = recommended["memory_fraction"]
        if "max_cache_entries" in recommended:
            config.cache.max_entries = recommended["max_cache_entries"]
    
    return config

if __name__ == "__main__":
    # Exemplo de uso
    config = get_config_for_environment("rtx_2060_optimized")
    
    print("=== Configuração Otimizada para RTX 2060 ===")
    optimized = config.get_optimized_config_for_rtx2060()
    print(json.dumps(optimized, indent=2, ensure_ascii=False))
    
    print("/n=== Validação da Configuração ===")
    validation = config.validate_config()
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    
    # Salvar configuração
    config_file = PROJECT_ROOT / "config" / "rag_optimized.json"
    config.save_to_file(str(config_file))
    print(f"/nConfiguração salva em: {config_file}")