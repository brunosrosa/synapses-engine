# -*- coding: utf-8 -*-
"""
Gerenciador de Servidor RAG para Recoloca.ai

Este módulo implementa o padrão de injeção de dependências e gerenciamento
de lifecycle para o sistema RAG, eliminando singletons globais problemáticos.

Autor: @AgenteM_DevFastAPI
Versão: 2.0 (Refatoração)
Data: Janeiro 2025
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config_manager import RAGConfig, ConfigManager

# Configurar logging
logger = logging.getLogger(__name__)

# =============================================================================
# PROTOCOLOS E INTERFACES
# =============================================================================

class RAGComponent(Protocol):
    """Protocol para componentes RAG gerenciáveis."""
    
    async def initialize(self, config: RAGConfig) -> None:
        """Inicializa o componente."""
        ...
    
    async def cleanup(self) -> None:
        """Limpa recursos do componente."""
        ...
    
    def is_healthy(self) -> bool:
        """Verifica se o componente está saudável."""
        ...

T = TypeVar('T', bound=RAGComponent)

class ComponentFactory(Protocol, Generic[T]):
    """Factory para criação de componentes RAG."""
    
    def create(self, config: RAGConfig) -> T:
        """Cria uma instância do componente."""
        ...

# =============================================================================
# GERENCIAMENTO DE DEPENDÊNCIAS
# =============================================================================

@dataclass
class ComponentInfo:
    """Informações sobre um componente registrado."""
    name: str
    factory: ComponentFactory
    instance: Optional[RAGComponent] = None
    initialized: bool = False
    dependencies: List[str] = field(default_factory=list)
    initialization_time: Optional[float] = None
    last_health_check: Optional[datetime] = None
    health_status: bool = True

class DependencyInjector:
    """Injetor de dependências para componentes RAG."""
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._initialization_order: List[str] = []
        self._config: Optional[RAGConfig] = None
    
    def register_component(
        self,
        name: str,
        factory: ComponentFactory,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Registra um componente no injetor."""
        if name in self._components:
            raise ValueError(f"Componente '{name}' já registrado")
        
        self._components[name] = ComponentInfo(
            name=name,
            factory=factory,
            dependencies=dependencies or []
        )
        
        logger.debug(f"Componente '{name}' registrado com dependências: {dependencies}")
    
    def get_component(self, name: str) -> Optional[RAGComponent]:
        """Obtém uma instância de componente."""
        component_info = self._components.get(name)
        if not component_info:
            raise ValueError(f"Componente '{name}' não registrado")
        
        return component_info.instance
    
    def _resolve_initialization_order(self) -> List[str]:
        """Resolve a ordem de inicialização baseada nas dependências."""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Dependência circular detectada envolvendo '{component_name}'")
            
            if component_name not in visited:
                temp_visited.add(component_name)
                
                component_info = self._components.get(component_name)
                if component_info:
                    for dependency in component_info.dependencies:
                        if dependency not in self._components:
                            raise ValueError(f"Dependência '{dependency}' não registrada para '{component_name}'")
                        visit(dependency)
                
                temp_visited.remove(component_name)
                visited.add(component_name)
                order.append(component_name)
        
        for component_name in self._components:
            visit(component_name)
        
        return order
    
    async def initialize_all(self, config: RAGConfig) -> None:
        """Inicializa todos os componentes na ordem correta."""
        self._config = config
        self._initialization_order = self._resolve_initialization_order()
        
        logger.info(f"Inicializando componentes na ordem: {self._initialization_order}")
        
        for component_name in self._initialization_order:
            await self._initialize_component(component_name)
    
    async def _initialize_component(self, name: str) -> None:
        """Inicializa um componente específico."""
        component_info = self._components[name]
        
        if component_info.initialized:
            logger.debug(f"Componente '{name}' já inicializado")
            return
        
        logger.info(f"Inicializando componente '{name}'...")
        start_time = time.time()
        
        try:
            # Criar instância
            component_info.instance = component_info.factory.create(self._config)
            
            # Inicializar
            await component_info.instance.initialize(self._config)
            
            # Marcar como inicializado
            component_info.initialized = True
            component_info.initialization_time = time.time() - start_time
            
            logger.info(f"Componente '{name}' inicializado em {component_info.initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar componente '{name}': {e}")
            raise
    
    async def cleanup_all(self) -> None:
        """Limpa todos os componentes na ordem reversa."""
        cleanup_order = list(reversed(self._initialization_order))
        
        logger.info(f"Limpando componentes na ordem: {cleanup_order}")
        
        for component_name in cleanup_order:
            await self._cleanup_component(component_name)
    
    async def _cleanup_component(self, name: str) -> None:
        """Limpa um componente específico."""
        component_info = self._components[name]
        
        if not component_info.initialized or not component_info.instance:
            return
        
        logger.info(f"Limpando componente '{name}'...")
        
        try:
            await component_info.instance.cleanup()
            component_info.initialized = False
            component_info.instance = None
            
            logger.info(f"Componente '{name}' limpo com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao limpar componente '{name}': {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Executa health check em todos os componentes."""
        results = {}
        overall_health = True
        
        for name, component_info in self._components.items():
            if component_info.initialized and component_info.instance:
                try:
                    health = component_info.instance.is_healthy()
                    component_info.health_status = health
                    component_info.last_health_check = datetime.now()
                    
                    results[name] = {
                        "healthy": health,
                        "initialized": True,
                        "initialization_time": component_info.initialization_time,
                        "last_check": component_info.last_health_check.isoformat()
                    }
                    
                    if not health:
                        overall_health = False
                        
                except Exception as e:
                    logger.error(f"Erro no health check do componente '{name}': {e}")
                    results[name] = {
                        "healthy": False,
                        "initialized": True,
                        "error": str(e)
                    }
                    overall_health = False
            else:
                results[name] = {
                    "healthy": False,
                    "initialized": False
                }
                overall_health = False
        
        return {
            "overall_healthy": overall_health,
            "components": results,
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# GERENCIADOR PRINCIPAL DO SERVIDOR RAG
# =============================================================================

class RAGServerManager:
    """Gerenciador principal do servidor RAG com injeção de dependências."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or ConfigManager.get_config()
        self.injector = DependencyInjector()
        self._initialized = False
        self._startup_time: Optional[float] = None
        
        # Configurar logging estruturado
        self._setup_logging()
        
        # Registrar componentes padrão
        self._register_default_components()
    
    def _setup_logging(self) -> None:
        """Configura logging estruturado."""
        log_config = self.config.logging
        
        # Configurar nível de logging
        logging.getLogger().setLevel(getattr(logging, log_config.level))
        
        # Configurar formato
        formatter = logging.Formatter(
            log_config.format,
            datefmt=log_config.date_format
        )
        
        # Configurar handler de console se habilitado
        if log_config.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
        
        # Configurar handler de arquivo se habilitado
        if log_config.enable_file_logging and log_config.file_path:
            log_config.file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_config.file_path)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        
        logger.info("Logging configurado")
    
    def _register_default_components(self) -> None:
        """Registra componentes padrão do sistema."""
        # TODO: Implementar factories para componentes específicos
        # Exemplo:
        # self.injector.register_component(
        #     "embedding_model",
        #     EmbeddingModelFactory(),
        #     dependencies=[]
        # )
        # self.injector.register_component(
        #     "rag_retriever",
        #     RAGRetrieverFactory(),
        #     dependencies=["embedding_model"]
        # )
        
        logger.debug("Componentes padrão registrados")
    
    def register_component(
        self,
        name: str,
        factory: ComponentFactory,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Registra um componente customizado."""
        self.injector.register_component(name, factory, dependencies)
    
    def get_component(self, name: str) -> Optional[RAGComponent]:
        """Obtém um componente registrado."""
        return self.injector.get_component(name)
    
    async def initialize(self) -> None:
        """Inicializa o servidor RAG."""
        if self._initialized:
            logger.warning("Servidor RAG já inicializado")
            return
        
        logger.info("Inicializando servidor RAG...")
        start_time = time.time()
        
        try:
            # Validar configuração
            validation_results = ConfigManager.validate_environment()
            logger.info(f"Validação do ambiente: {validation_results}")
            
            # Inicializar componentes
            await self.injector.initialize_all(self.config)
            
            self._initialized = True
            self._startup_time = time.time() - start_time
            
            logger.info(f"Servidor RAG inicializado em {self._startup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Erro na inicialização do servidor RAG: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Limpa recursos do servidor RAG."""
        if not self._initialized:
            return
        
        logger.info("Limpando servidor RAG...")
        
        try:
            await self.injector.cleanup_all()
            self._initialized = False
            
            logger.info("Servidor RAG limpo com sucesso")
            
        except Exception as e:
            logger.error(f"Erro na limpeza do servidor RAG: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Executa health check completo do servidor."""
        component_health = self.injector.health_check()
        config_status = self.config.get_status_summary()
        
        return {
            "server": {
                "initialized": self._initialized,
                "startup_time": self._startup_time,
                "uptime": time.time() - (time.time() - (self._startup_time or 0))
            },
            "config": config_status,
            "components": component_health,
            "timestamp": datetime.now().isoformat()
        }
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager para gerenciar lifecycle do servidor."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtém métricas do servidor."""
        health = self.health_check()
        
        return {
            "server_metrics": {
                "initialized": self._initialized,
                "startup_time_seconds": self._startup_time,
                "components_count": len(self.injector._components),
                "healthy_components": sum(
                    1 for comp in health["components"]["components"].values()
                    if comp.get("healthy", False)
                )
            },
            "config_metrics": {
                "gpu_enabled": self.config.embedding.device == "cuda",
                "pytorch_backend": self.config.force_pytorch,
                "optimizations_enabled": self.config.use_optimizations
            },
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# UTILITÁRIOS E EXPORTS
# =============================================================================

def create_server_manager(config: Optional[RAGConfig] = None) -> RAGServerManager:
    """Factory function para criar um RAGServerManager."""
    return RAGServerManager(config)

# Exports
__all__ = [
    "RAGServerManager",
    "DependencyInjector",
    "RAGComponent",
    "ComponentFactory",
    "ComponentInfo",
    "create_server_manager"
]