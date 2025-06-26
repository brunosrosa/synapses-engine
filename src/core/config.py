"""Configurações centralizadas para o sistema RAG."""

from pathlib import Path
import os
from typing import Dict, Any, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Diretório base do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAG_INFRA_ROOT = Path(__file__).resolve().parents[2]

# Diretórios principais (nova estrutura)
RESULTS_AND_REPORTS_DIR = RAG_INFRA_ROOT / "reports" 
LOGS_DIR = RAG_INFRA_ROOT / "logs" / "application"
DATA_INDEX_DIR = RAG_INFRA_ROOT / "data" / "indexes"
SOURCE_DOCUMENTS_DIR = RAG_INFRA_ROOT / "knowledge_base_for_rag"
PYTORCH_INDEX_DIR = DATA_INDEX_DIR / "pytorch_index_bge_m3"

# Diretórios organizacionais (nova estrutura)
DIAGNOSTICS_DIR = RAG_INFRA_ROOT / "src" / "tests" / "diagnostics"
SERVER_DIR = RAG_INFRA_ROOT / "server"
SETUP_DIR = RAG_INFRA_ROOT / "setup"
TESTS_DIR = RAG_INFRA_ROOT / "src" / "tests"
SCRIPTS_DIR = RAG_INFRA_ROOT / "scripts"
CORE_LOGIC_DIR = RAG_INFRA_ROOT / "src" / "core"
UTILS_DIR = RAG_INFRA_ROOT / "src" / "utils"

# Garantir que os diretórios existam
RESULTS_AND_REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
SERVER_DIR.mkdir(exist_ok=True)
SETUP_DIR.mkdir(exist_ok=True)

# Funções utilitárias para caminhos de relatórios
def get_report_path(report_name: str) -> Path:
    """Retorna o caminho completo para um arquivo de relatório.
    
    Args:
        report_name: Nome do arquivo de relatório (ex: 'rag_test_report.json')
        
    Returns:
        Path: Caminho completo para o arquivo no diretório results_and_reports
    """
    if not report_name.endswith('.json'):
        report_name += '.json'
    return RESULTS_AND_REPORTS_DIR / report_name

def get_log_path(log_name: str) -> Path:
    """Retorna o caminho completo para um arquivo de log.
    
    Args:
        log_name: Nome do arquivo de log (ex: 'rag_test.log')
        
    Returns:
        Path: Caminho completo para o arquivo no diretório logs
    """
    if not log_name.endswith('.log'):
        log_name += '.log'
    return LOGS_DIR / log_name

# Funções utilitárias para módulos reorganizados
def get_module_path(module_category: str, module_name: str = None) -> Path:
    """Retorna o caminho para um módulo específico na estrutura reorganizada.
    
    Args:
        module_category: Categoria do módulo ('diagnostics', 'server', 'setup', etc.)
        module_name: Nome específico do módulo (opcional)
        
    Returns:
        Path: Caminho para o diretório ou arquivo do módulo
    """
    category_dirs = {
        'diagnostics': DIAGNOSTICS_DIR,
        'server': SERVER_DIR,
        'setup': SETUP_DIR,
        'tests': TESTS_DIR,
        'scripts': SCRIPTS_DIR,
        'core': CORE_LOGIC_DIR,
        'utils': UTILS_DIR
    }
    
    base_dir = category_dirs.get(module_category)
    if not base_dir:
        raise ValueError(f"Categoria de módulo desconhecida: {module_category}")
    
    if module_name:
        if not module_name.endswith('.py'):
            module_name += '.py'
        return base_dir / module_name
    
    return base_dir

def add_module_to_path(module_category: str):
    """Adiciona um diretório de módulo ao sys.path para facilitar imports.
    
    Args:
        module_category: Categoria do módulo a ser adicionada ao path
    """
    import sys
    module_dir = get_module_path(module_category)
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

# Mapeamento de arquivos reorganizados (para compatibilidade)
REORGANIZED_FILES = {
    'correcao_rag.py': 'src/tests/diagnostics/correcao_rag.py',
    'diagnostico_rag.py': 'src/tests/diagnostics/diagnostico_rag.py',
    'diagnostico_simples.py': 'src/tests/diagnostics/diagnostico_simples.py',
    'mcp_server.py': 'server/mcp_server.py',
    'setup_rag.py': 'setup/setup_rag.py',
    'test_rag_quick.py': 'src/tests/test_rag_quick.py'
}



class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='RAG_LOG_', extra='ignore')
    log_level: str = "DEBUG"
    log_file: Optional[Path] = Field(
        default_factory=lambda: Path(os.getenv('RAG_LOG_FILE')) if os.getenv('RAG_LOG_FILE') else None
    )
    log_max_bytes: int = 10485760
    log_backup_count: int = 5
    log_format: str = "{time} {level} {message}"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    log_use_colors: bool = True
    log_rotation_interval: int = 1
    log_rotation_unit: str = "day"
    log_compression: Optional[str] = Field(
        default_factory=lambda: os.getenv('RAG_LOG_COMPRESSION')
    )
    log_retention: Optional[str] = Field(
        default_factory=lambda: os.getenv('RAG_LOG_RETENTION')
    )
    log_encoding: str = "utf-8"
    log_diagnostic: bool = False
    log_serialize: bool = False
    log_backtrace: bool = False
    log_catch_exceptions: bool = False
    log_quiet: bool = False
    log_colorize: bool = True
    log_levels_colors: Optional[str] = None



class EmbeddingModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='RAG_EMBEDDING_MODEL_', extra='ignore')
    name: str = "BAAI/bge-m3"
    max_tokens: int = 8192
    device: str = "auto"
    trust_remote_code: bool = False
    revision: Optional[str] = None
    token: Optional[str] = None
    cache_dir: Optional[Path] = None
    use_langchain_embeddings: bool = Field(False)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"normalize_embeddings": True})



class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='RAG_', extra='ignore')

    # Sub-configurações
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Diretórios
    data_index_dir: Path = DATA_INDEX_DIR
    source_documents_dir: Path = SOURCE_DOCUMENTS_DIR
    results_and_reports_dir: Path = RESULTS_AND_REPORTS_DIR
    logs_dir: Path = LOGS_DIR

    # Parâmetros do Sentence Transformer
    sentence_transformer_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "device": "cpu",
            "cache_dir": None,
            "trust_remote_code": False,
            "revision": None,
            "token": None,
        }
    )

    # Parâmetros de busca RAG
    top_k_default: int = Field(5, description="Número padrão de resultados a retornar na busca RAG.")
    similarity_threshold: float = Field(0.2, description="Limiar de similaridade para filtragem de resultados RAG.")
    index_path: Optional[Path] = Field(None, description="Caminho para o índice FAISS persistido.")

    # Otimizações e Performance
    force_cpu: bool = Field(False, description="Força o uso da CPU para operações de embedding e FAISS.")
    force_pytorch: bool = Field(False, description="Força o uso do PyTorch para operações de embedding e FAISS.")
    use_optimizations: bool = Field(True, description="Habilita otimizações de performance (ex: quantização, compilação JIT).")
    rag_cache_max_size: int = Field(100, description="Tamanho máximo do cache de resultados RAG em MB.")
    rag_use_gpu: bool = Field(False, description="Define se o RAG deve tentar usar GPU (se disponível).")
    pytorch_batch_size: int = Field(32, description="Tamanho do batch para inferência PyTorch.")
    pytorch_index_dir: Path = Field(PYTORCH_INDEX_DIR, description="Diretório para índices PyTorch.")

    # Mensagens de Status e Templates
    status_messages: Dict[str, str] = Field(
        default_factory=lambda: {
            "indexing_start": "Iniciando a indexação dos documentos...",
            "indexing_complete": "Indexação concluída. {num_docs} documentos processados.",
            "search_start": "Realizando busca RAG para a consulta: '{query}'",
            "search_results": "{num_results} resultados encontrados.",
            "no_results": "Nenhum resultado relevante encontrado para a consulta.",
        }
    )
    result_template: str = Field(
        "**Documento:** {doc_id}\n**Score:** {score:.2f}\n**Conteúdo:** {content}\n---",
        description="Template para formatar os resultados da busca RAG."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Converte a configuração para um dicionário, incluindo sub-configurações."""
        return self.model_dump(mode='json')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Cria uma instância de RAGConfig a partir de um dicionário."""
        return cls.model_validate(data)