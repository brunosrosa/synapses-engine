#!/usr/bin/env python3
"""
Script de Setup do Sistema RAG - Recoloca.ai

Este script automatiza a configuração inicial do sistema RAG, incluindo:
- Verificação de dependências
- Configuração do ambiente
- Indexação inicial dos documentos
- Teste do sistema

Uso:
    python setup_rag.py [--force-cpu] [--skip-indexing] [--verbose]
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import datetime
import json

# Adicionar o diretório src ao path para permitir importações absolutas
# __file__ -> .../rag_infra/src/core/setup_rag.py
# .parent -> .../rag_infra/src/core
# .parent.parent -> .../rag_infra/src
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from core.config_manager import RAGConfig
from core.embedding_model import initialize_embedding_model
from core.rag_indexer import RAGIndexer
from core.rag_retriever import RAGRetriever

class RAGSetup:
    """Classe para gerenciar o setup do sistema RAG."""
    
    def __init__(self, config: RAGConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.setup_logging()
        
    def setup_logging(self):
        """Configura o sistema de logging."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.logs_dir / "setup_rag.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_python_version(self) -> bool:
        """Verifica se a versão do Python é compatível."""
        self.logger.info("[EMOJI] Verificando versão do Python...")
        
        if sys.version_info < (3, 8):
            self.logger.error("[ERROR] Python 3.8+ é necessário")
            return False
            
        self.logger.info(f"[OK] Python {sys.version.split()[0]} detectado")
        return True
        
    def check_gpu_availability(self) -> Tuple[bool, str]:
        """Verifica disponibilidade de GPU e CUDA."""
        self.logger.info(self.config.status_messages["checking_gpu"])
        
        if self.config.force_cpu:
            self.logger.info(self.config.status_messages["force_cpu_enabled"])
            return False, "CPU forçada pelo usuário"
            
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(self.config.status_messages["gpu_detected"].format(gpu_name=gpu_name, gpu_memory=gpu_memory))
                return True, f"{gpu_name} ({gpu_memory:.1f}GB)"
            else:
                self.logger.warning(self.config.status_messages["cuda_unavailable"])
                return False, "CUDA não disponível"
        except ImportError:
            self.logger.warning(self.config.status_messages["pytorch_not_installed"])
            return False, "PyTorch não instalado"
            
    def check_dependencies(self) -> List[str]:
        """Verifica se todas as dependências estão instaladas."""
        self.logger.info("[EMOJI] Verificando dependências...")
        
        # Mapeamento de pacotes pip para nomes de importação
        required_packages = {
            'torch': 'torch',
            'transformers': 'transformers', 
            'sentence-transformers': 'sentence_transformers',
            'langchain': 'langchain',
            'faiss-cpu': 'faiss',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'pymupdf': 'fitz',
            'python-dotenv': 'dotenv',
            'unstructured': 'unstructured'
        }
        
        missing_packages = []
        
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                self.logger.debug(f"[OK] {package_name} instalado")
            except ImportError:
                missing_packages.append(package_name)
                self.logger.warning(f"[ERROR] {package_name} não encontrado")
                
        if missing_packages:
            self.logger.error(f"[ERROR] Pacotes faltando: {', '.join(missing_packages)}")
        else:
            self.logger.info("[OK] Todas as dependências estão instaladas")
            
        return missing_packages
        
    def create_directories(self) -> bool:
        """Cria os diretórios necessários."""
        self.logger.info(self.config.status_messages["creating_directories"])
        
        directories = [
            self.config.rag_root_dir,
            self.config.source_documents_dir,
            self.config.faiss_index_dir,
            self.config.logs_dir
        ]
        
        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(self.config.status_messages["directory_created"].format(directory=directory))
                
            self.logger.info(self.config.status_messages["directories_created"])
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Erro ao criar diretórios: {e}")
            return False
            
    def check_source_documents(self) -> Tuple[bool, int]:
        """Verifica se há documentos para indexar."""
        self.logger.info(self.config.status_messages["checking_source_documents"])
        
        supported_extensions = {'.md', '.txt', '.pdf', '.docx', '.html'}
        document_count = 0
        
        for ext in supported_extensions:
            files = list(self.config.source_documents_dir.rglob(f"*{ext}"))
            document_count += len(files)
            if files:
                self.logger.debug(self.config.status_messages["files_found"].format(count=len(files), ext=ext))
                
        if document_count == 0:
            self.logger.warning(self.config.status_messages["no_documents_found"])
            self.logger.info(self.config.status_messages["add_documents_prompt"].format(source_dir=self.config.source_documents_dir))
            return False, 0
        else:
            self.logger.info(self.config.status_messages["documents_found"].format(count=document_count))
            return True, document_count
            
    def test_embedding_model(self) -> bool:
        """Testa o modelo de embedding."""
        self.logger.info(self.config.status_messages["testing_embedding_model"])
        
        try:
            embedding_manager = initialize_embedding_model(config=self.config)
            
            test_phrase = self.config.embedding_model_test_phrase
            embedding = embedding_manager.embed_query(test_phrase)
            
            if embedding is not None and len(embedding) > 0:
                self.logger.info(self.config.status_messages["embedding_model_test_success"].format(dimension=len(embedding)))
                embedding_manager.unload_model()
                return True
            else:
                self.logger.error(self.config.status_messages["embedding_model_test_failure"])
                embedding_manager.unload_model()
                return False
        except Exception as e:
            self.logger.error(self.config.status_messages["embedding_model_test_error"].format(error=e))
            return False
            
    def run_indexing(self) -> bool:
        """Executa a indexação dos documentos."""
        self.logger.info("[EMOJI] Iniciando indexação dos documentos...")
        
        try:
            indexer = RAGIndexer(config=self.config)
            success = indexer.index_documents()
            
            if success:
                self.logger.info("[OK] Indexação concluída com sucesso")
                return True
            else:
                self.logger.error("[ERROR] Falha na indexação")
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Erro durante indexação: {e}")
            return False
            
    def test_retrieval(self) -> bool:
        """Testa o sistema de recuperação."""
        self.logger.info("[SEARCH] Testando sistema de recuperação...")
        
        try:
            retriever = RAGRetriever(config=self.config)
            
            # Carregar o índice primeiro
            if not retriever.load_index():
                self.logger.error("Falha ao carregar o índice para o teste de recuperação.")
                return False
            
            # Teste de consulta
            test_query = "arquitetura do sistema"
            results = retriever.search(test_query, top_k=3)
            
            if results:
                self.logger.info(f"[OK] Recuperação funcionando: {len(results)} resultados")
                for i, result in enumerate(results[:2], 1):
                    self.logger.info(f"  {i}. {result.document_name} (score: {result.score:.3f})")
                return True
            else:
                self.logger.warning("[WARNING] Nenhum resultado encontrado")
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Erro no teste de recuperação: {e}")
            return False
            
    def test_mcp_server(self) -> bool:
        """Testa o servidor MCP."""
        self.logger.info("[EMOJI] Testando servidor MCP...")
        
        try:
            # Importar e testar o servidor MCP
            mcp_server_path = RAG_ROOT_DIR / "mcp_server.py"
            if not mcp_server_path.exists():
                self.logger.error("[ERROR] Arquivo mcp_server.py não encontrado")
                return False
                
            # Teste básico de importação
            spec = subprocess.run(
                [sys.executable, "-c", "import sys; sys.path.insert(0, 'rag_infra'); import mcp_server"],
                cwd=RAG_ROOT_DIR.parent,
                capture_output=True,
                text=True
            )
            
            if spec.returncode == 0:
                self.logger.info("[OK] Servidor MCP pode ser importado")
                return True
            else:
                self.logger.error(f"[ERROR] Erro ao importar MCP server: {spec.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"[ERROR] Erro no teste do MCP server: {e}")
            return False
            
    def generate_report(self, results: dict) -> str:
        """Gera relatório do setup."""
        report = []
        report.append("\n" + "="*60)
        report.append("[EMOJI] RELATÓRIO DE SETUP DO SISTEMA RAG")
        report.append("="*60)
        
        # Status geral
        total_checks = len(results)
        passed_checks = sum(1 for v in results.values() if v)
        success_rate = (passed_checks / total_checks) * 100
        
        report.append(f"\n[EMOJI] Status Geral: {passed_checks}/{total_checks} verificações passaram ({success_rate:.1f}%)")
        
        # Detalhes das verificações
        report.append("\n[EMOJI] Detalhes das Verificações:")
        for check, status in results.items():
            icon = "[OK]" if status else "[ERROR]"
            report.append(f"  {icon} {check}")
            
        # Próximos passos
        report.append("\n[START] Próximos Passos:")
        
        if all(results.values()):
            report.append("  [OK] Sistema RAG está pronto para uso!")
            report.append("  [EMOJI] Configure o Trae IDE usando: config/trae_mcp_config.json")
            report.append("  [EMOJI] Execute consultas usando o MCP Server")
        else:
            report.append("  [WARNING] Corrija os problemas identificados acima")
            report.append("  [EMOJI] Execute o setup novamente após as correções")
            
        # Comandos úteis
        report.append("\n[EMOJI] Comandos Úteis:")
        report.append("  • Reindexar: python rag_indexer.py")
        report.append("  • Testar consulta: python rag_retriever.py")
        report.append("  • Iniciar MCP Server: python mcp_server.py")
        report.append("  • Executar testes: python tests/test_rag_system.py")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
        
    def run_setup(self, skip_indexing: bool = False) -> bool:
        """Executa o setup completo do sistema RAG."""
        self.logger.info("[START] Iniciando setup do Sistema RAG - Recoloca.ai")
        
        # Dicionário para armazenar resultados das verificações
        results = {}
        
        # 1. Verificar versão do Python
        results["Python 3.8+"] = self.check_python_version()
        
        # 2. Verificar GPU
        gpu_available, gpu_info = self.check_gpu_availability()
        results[f"GPU ({gpu_info})"] = gpu_available or self.config.force_cpu
        
        # 3. Verificar dependências
        missing_deps = self.check_dependencies()
        results["Dependências"] = len(missing_deps) == 0
        
        # 4. Criar diretórios
        results["Estrutura de diretórios"] = self.create_directories()
        
        # 5. Verificar documentos fonte
        has_docs, doc_count = self.check_source_documents()
        results[f"Documentos fonte ({doc_count})"] = has_docs
        
        # 6. Testar modelo de embedding
        results["Modelo de embedding"] = self.test_embedding_model()
        
        # 7. Executar indexação (se solicitado)
        if not skip_indexing and has_docs:
            results["Indexação"] = self.run_indexing()
            
            # 8. Testar recuperação
            if results.get("Indexação", False):
                results["Sistema de recuperação"] = self.test_retrieval()
        else:
            self.logger.info("⏭[EMOJI] Indexação pulada")
            
        # 9. Testar MCP Server
        results["Servidor MCP"] = self.test_mcp_server()
        
        # Gerar e exibir relatório
        report = self.generate_report(results)
        print(report)
        
        # Salvar relatório
        report_file = self.config.logs_dir / "setup_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # Gerar relatório de diagnóstico em JSON
        diagnostic_report_path = self.config.logs_dir / f"diagnostic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(diagnostic_report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"[EMOJI] Relatório de diagnóstico JSON salvo em: {diagnostic_report_path}")
        self.logger.info(f"[EMOJI] Relatório salvo em: {report_file}")
        
        # Retornar sucesso geral
        return all(results.values())

def main():
    """Função principal do script de setup."""
    parser = argparse.ArgumentParser(
        description="Setup do Sistema RAG - Recoloca.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python setup_rag.py                    # Setup completo
  python setup_rag.py --force-cpu        # Forçar uso de CPU
  python setup_rag.py --skip-indexing    # Pular indexação
  python setup_rag.py --verbose          # Modo verboso
        """
    )
    
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Forçar uso de CPU mesmo se GPU estiver disponível'
    )
    
    parser.add_argument(
        '--skip-indexing',
        action='store_true',
        help='Pular a etapa de indexação dos documentos'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Ativar modo verboso (debug)'
    )
    
    args = parser.parse_args()

    # Criar instância de RAGConfig
    config = RAGConfig(
        embedding_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        force_cpu=args.force_cpu,
        verbose=args.verbose
    )
    
    # Executar setup
    setup = RAGSetup(config=config)
    success = setup.run_setup(skip_indexing=args.skip_indexing)
    
    # Exit code baseado no sucesso
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()