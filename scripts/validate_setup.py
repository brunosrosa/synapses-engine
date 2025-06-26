#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Validação do Setup RAG - Recoloca.ai

Este script valida a configuração completa do sistema RAG, incluindo:
- Configuração GPU/PyTorch
- Validação de dependências
- Teste de componentes principais
- Verificação de performance

Autor: @AgenteM_DevFastAPI
Versão: 2.0 (Refatoração)
Data: Janeiro 2025

Uso:
    python scripts/validate_setup.py
    python scripts/validate_setup.py --verbose
    python scripts/validate_setup.py --gpu-test
"""

import sys
import os
import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.config_manager import ConfigManager, RAGConfig
    from core.server_manager import RAGServerManager
except ImportError as e:
    print(f"❌ Erro ao importar módulos do projeto: {e}")
    print("Certifique-se de que está executando do diretório correto")
    sys.exit(1)

def print_header(title: str) -> None:
    """Imprime um cabeçalho formatado."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(test_name: str, success: bool, details: str = "") -> None:
    """Imprime o resultado de um teste."""
    status = "✅ PASSOU" if success else "❌ FALHOU"
    print(f"{test_name:<40} {status}")
    if details:
        print(f"   {details}")

def validate_python_environment() -> Tuple[bool, str]:
    """Valida o ambiente Python."""
    try:
        python_version = sys.version_info
        if python_version < (3, 8):
            return False, f"Python {python_version.major}.{python_version.minor} não suportado (mínimo: 3.8)"
        
        return True, f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
    except Exception as e:
        return False, str(e)

def validate_dependencies() -> Tuple[bool, List[str]]:
    """Valida as dependências principais."""
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("numpy", "NumPy"),
        ("pydantic", "Pydantic"),
        ("pydantic_settings", "Pydantic Settings"),
        ("psutil", "PSUtil")
    ]
    
    missing = []
    details = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            details.append(f"✅ {name}")
        except ImportError:
            missing.append(name)
            details.append(f"❌ {name}")
    
    return len(missing) == 0, details

def validate_pytorch_gpu() -> Tuple[bool, str]:
    """Valida a configuração PyTorch/GPU."""
    try:
        import torch
        
        details = []
        details.append(f"PyTorch versão: {torch.__version__}")
        details.append(f"CUDA disponível: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            details.append(f"Dispositivos CUDA: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                details.append(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return True, "\n   ".join(details)
    except Exception as e:
        return False, str(e)

def validate_config_manager() -> Tuple[bool, str]:
    """Valida o ConfigManager."""
    try:
        # Testar criação de configuração
        config = ConfigManager.get_config()
        
        details = []
        details.append(f"Configuração carregada: {type(config).__name__}")
        details.append(f"Device: {config.embedding.device}")
        details.append(f"GPU habilitado: {config.embedding.device == 'cuda'}")
        
        # Testar validação do ambiente
        validation_results = ConfigManager.validate_environment()
        details.append(f"Validação do ambiente: {validation_results['overall_status']}")
        
        return True, "\n   ".join(details)
    except Exception as e:
        return False, str(e)

def validate_server_manager() -> Tuple[bool, str]:
    """Valida o RAGServerManager."""
    try:
        # Testar criação do servidor
        server = RAGServerManager()
        
        details = []
        details.append(f"Servidor criado: {type(server).__name__}")
        details.append(f"Configuração válida: {server.config is not None}")
        
        # Testar health check básico
        health = server.health_check()
        details.append(f"Health check executado: {health is not None}")
        
        return True, "\n   ".join(details)
    except Exception as e:
        return False, str(e)

def run_gpu_performance_test() -> Tuple[bool, str]:
    """Executa teste de performance da GPU."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "CUDA não disponível"
        
        device = torch.device("cuda")
        
        # Teste básico de operações
        start_time = time.time()
        
        # Criar tensores de teste
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Operação matricial
        z = torch.matmul(x, y)
        
        # Sincronizar GPU
        torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        # Verificar memória
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        memory_cached = torch.cuda.memory_reserved(device) / 1024**2
        
        details = [
            f"Operação matricial (1000x1000): {elapsed:.3f}s",
            f"Memória alocada: {memory_allocated:.1f}MB",
            f"Memória em cache: {memory_cached:.1f}MB"
        ]
        
        # Limpar memória
        del x, y, z
        torch.cuda.empty_cache()
        
        return True, "\n   ".join(details)
    except Exception as e:
        return False, str(e)

def run_embedding_test() -> Tuple[bool, str]:
    """Testa o modelo de embedding básico."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Testar carregamento do modelo
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        start_time = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        
        # Testar encoding
        test_texts = ["Este é um teste", "Outro texto de exemplo"]
        
        start_time = time.time()
        embeddings = model.encode(test_texts)
        encode_time = time.time() - start_time
        
        details = [
            f"Modelo carregado em: {load_time:.2f}s",
            f"Encoding de 2 textos: {encode_time:.3f}s",
            f"Dimensão dos embeddings: {embeddings.shape[1]}",
            f"Device do modelo: {model.device}"
        ]
        
        return True, "\n   ".join(details)
    except Exception as e:
        return False, str(e)

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Validação do Setup RAG")
    parser.add_argument("--verbose", "-v", action="store_true", help="Saída detalhada")
    parser.add_argument("--gpu-test", "-g", action="store_true", help="Executar testes de GPU")
    parser.add_argument("--embedding-test", "-e", action="store_true", help="Testar modelo de embedding")
    
    args = parser.parse_args()
    
    print_header("VALIDAÇÃO DO SETUP RAG - RECOLOCA.AI")
    
    tests = [
        ("Ambiente Python", validate_python_environment),
        ("Dependências", validate_dependencies),
        ("PyTorch/GPU", validate_pytorch_gpu),
        ("ConfigManager", validate_config_manager),
        ("RAGServerManager", validate_server_manager)
    ]
    
    if args.gpu_test:
        tests.append(("Performance GPU", run_gpu_performance_test))
    
    if args.embedding_test:
        tests.append(("Modelo de Embedding", run_embedding_test))
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success, details = test_func()
            results.append((test_name, success, details))
            
            if args.verbose or not success:
                print_result(test_name, success, details)
            else:
                print_result(test_name, success)
                
        except Exception as e:
            results.append((test_name, False, str(e)))
            print_result(test_name, False, f"Erro: {e}")
            if args.verbose:
                traceback.print_exc()
    
    # Resumo final
    print_header("RESUMO DOS TESTES")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Testes executados: {total}")
    print(f"Testes aprovados: {passed}")
    print(f"Testes falharam: {total - passed}")
    
    if passed == total:
        print("\n🎉 Todos os testes passaram! O sistema está pronto para uso.")
        return 0
    else:
        print("\n⚠️  Alguns testes falharam. Verifique a configuração.")
        return 1

if __name__ == "__main__":
    sys.exit(main())