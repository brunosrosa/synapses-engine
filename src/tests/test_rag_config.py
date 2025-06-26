import pytest
import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "rag_infra" / "src"))

from core.config import RAGConfig, LoggingConfig, EmbeddingModelConfig, LOGS_DIR

def test_rag_config_default_values():
    """Testa se a configuração carrega os valores padrão corretamente."""
    config = RAGConfig()
    assert config.embedding_model_config.name == "BAAI/bge-m3"
    assert config.embedding_model_config.max_tokens == 8192
    assert config.embedding_model_config.device == "auto"
    assert config.embedding_model_config.trust_remote_code is False
    assert config.embedding_model_config.revision is None
    assert config.embedding_model_config.token is None
    assert config.embedding_model_config.cache_dir is None
    assert config.logs_dir == LOGS_DIR

def test_rag_config_custom_instantiation():
    """
    Testa a configuração do RAG instanciando as classes com valores customizados,
    em vez de depender de variáveis de ambiente, garantindo um teste robusto.
    """
    logging_settings = {
        "log_level": "DEBUG",
        "log_file": Path("test_log.log"),
        "log_max_bytes": 10485760,
        "log_backup_count": 5,
        "log_format": "{time} {level} {message}",
        "log_date_format": "%Y-%m-%d %H:%M:%S",
        "log_use_colors": True,
        "log_rotation_interval": 1,
        "log_rotation_unit": "day",
        "log_compression": "zip",
        "log_retention": "1 week",
        "log_encoding": "utf-8",
        "log_diagnostic": True,
        "log_serialize": True,
        "log_backtrace": True,
        "log_catch_exceptions": True,
        "log_quiet": False,
        "log_colorize": True,
        "log_levels_colors": "INFO:green",
    }

    embedding_settings = {
        "name": "test_model",
        "max_tokens": 128,
        "device": "cpu",
        "trust_remote_code": True,
        "revision": "test_revision",
        "token": "test_token",
        "cache_dir": Path("/tmp/test_cache"),
    }

    config = RAGConfig(
        logs_dir=Path("/tmp/test_rag_logs"),
        logging_config=LoggingConfig(**logging_settings),
        embedding_model_config=EmbeddingModelConfig(**embedding_settings),
    )

    # Asserções para EmbeddingModelConfig
    assert config.embedding_model_config.name == "test_model"
    assert config.embedding_model_config.max_tokens == 128
    assert config.embedding_model_config.device == "cpu"
    assert config.embedding_model_config.trust_remote_code is True
    assert config.embedding_model_config.revision == "test_revision"
    assert config.embedding_model_config.token == "test_token"
    assert config.embedding_model_config.cache_dir == Path("/tmp/test_cache")

    # Asserções para RAGConfig
    assert config.logs_dir == Path("/tmp/test_rag_logs")

    # Asserções para LoggingConfig
    log_config = config.logging_config
    assert log_config.log_level == "DEBUG"
    assert log_config.log_file == Path("test_log.log")
    assert log_config.log_max_bytes == 10485760
    assert log_config.log_backup_count == 5
    assert log_config.log_format == "{time} {level} {message}"
    assert log_config.log_date_format == "%Y-%m-%d %H:%M:%S"
    assert log_config.log_use_colors is True
    assert log_config.log_rotation_interval == 1
    assert log_config.log_rotation_unit == "day"
    assert log_config.log_compression == "zip"
    assert log_config.log_retention == "1 week"
    assert log_config.log_encoding == "utf-8"
    assert log_config.log_diagnostic is True
    assert log_config.log_serialize is True
    assert log_config.log_backtrace is True
    assert log_config.log_catch_exceptions is True
    assert log_config.log_quiet is False
    assert log_config.log_colorize is True
    assert log_config.log_levels_colors == "INFO:green"