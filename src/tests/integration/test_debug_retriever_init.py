import sys
import logging
from pathlib import Path

# Configurar logging para capturar tudo
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("Iniciando script de depuracao de inicializacao do RAGRetriever.")

# Adicionar o diretório raiz do projeto ao path para resolver importacoes
try:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_path = project_root / 'src'
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_path))
    logger.info(f"Project root adicionado ao path: {project_root}")
    logger.info(f"Source path adicionado ao path: {src_path}")
    logger.info(f"sys.path atual: {sys.path}")
except Exception as e:
    logger.critical(f"Falha ao configurar o sys.path: {e}", exc_info=True)
    sys.exit(1)

try:
    logger.debug("Tentando importar RAGRetriever de rag_infra.src.core.rag_retriever...")
    from rag_infra.src.core.rag_retriever import RAGRetriever
    logger.info("Importacao de RAGRetriever bem-sucedida.")
except Exception as e:
    logger.critical(f"Falha CRITICA ao importar RAGRetriever: {e}", exc_info=True)
    sys.exit(1)

try:
    logger.debug("Tentando instanciar RAGRetriever com configuracao basica (force_cpu=True)...")
    retriever = RAGRetriever(force_cpu=True)
    logger.info("Instanciacao de RAGRetriever bem-sucedida.")
except Exception as e:
    logger.critical(f"Falha CRITICA ao instanciar RAGRetriever: {e}", exc_info=True)
    sys.exit(1)

try:
    logger.debug("Tentando inicializar o retriever...")
    initialized = retriever.initialize()
    if initialized:
        logger.info("Inicializacao do RAGRetriever (metodo initialize()) concluida com sucesso.")
    else:
        logger.error("Metodo initialize() do RAGRetriever retornou False.")
except Exception as e:
    logger.critical(f"Falha CRITICA durante a chamada ao metodo initialize() do RAGRetriever: {e}", exc_info=True)
    sys.exit(1)

logger.info("Script de depuracao concluido com sucesso. A classe RAGRetriever pode ser importada, instanciada e inicializada.")