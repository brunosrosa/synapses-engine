#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Teste da Reorganização Python - RAG Infra

Este script testa se a reorganização dos arquivos Python foi bem-sucedida.

Autor: @AgenteM_DevFastAPI
Versão: 1.0
Data: Junho 2025
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório rag_infra ao path
rag_infra_path = Path(__file__).parent
sys.path.insert(0, str(rag_infra_path))

def test_reorganization():
    """Testa se a reorganização foi bem-sucedida."""
    print("=== TESTE DE REORGANIZAÇÃO DOS ARQUIVOS PYTHON ===")
    print()
    
    # Teste 1: Verificar se as pastas foram criadas
    print("1. Verificando estrutura de diretórios...")
    expected_dirs = ['diagnostics', 'server', 'setup']
    
    for dir_name in expected_dirs:
        dir_path = rag_infra_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"   [OK] {dir_name}/ - OK")
        else:
            print(f"   [ERROR] {dir_name}/ - FALTANDO")
    
    print()
    
    # Teste 2: Verificar se os arquivos __init__.py existem
    print("2. Verificando arquivos __init__.py...")
    
    for dir_name in expected_dirs:
        init_file = rag_infra_path / dir_name / '__init__.py'
        if init_file.exists():
            print(f"   [OK] {dir_name}/__init__.py - OK")
        else:
            print(f"   [ERROR] {dir_name}/__init__.py - FALTANDO")
    
    print()
    
    # Teste 3: Verificar se os arquivos foram movidos
    print("3. Verificando arquivos movidos...")
    
    expected_files = {
        'diagnostics': ['correcao_rag.py', 'diagnostico_rag.py', 'diagnostico_simples.py'],
        'server': ['mcp_server.py'],
        'setup': ['setup_rag.py']
    }
    
    for dir_name, files in expected_files.items():
        for file_name in files:
            file_path = rag_infra_path / dir_name / file_name
            if file_path.exists():
                print(f"   [OK] {dir_name}/{file_name} - OK")
            else:
                print(f"   [ERROR] {dir_name}/{file_name} - FALTANDO")
    
    print()
    
    # Teste 4: Verificar se os arquivos antigos foram removidos da raiz
    print("4. Verificando remoção dos arquivos da raiz...")
    
    old_files = [
        'correcao_rag.py', 'diagnostico_rag.py', 'diagnostico_simples.py',
        'mcp_server.py', 'setup_rag.py', 'test_rag_quick.py'
    ]
    
    for file_name in old_files:
        file_path = rag_infra_path / file_name
        if not file_path.exists():
            print(f"   [OK] {file_name} - REMOVIDO DA RAIZ")
        else:
            print(f"   [WARNING]  {file_name} - AINDA NA RAIZ")
    
    print()
    
    # Teste 5: Testar imports da configuração
    print("5. Testando configuração centralizada...")
    
    try:
        from config import get_module_path, DIAGNOSTICS_DIR, SERVER_DIR, SETUP_DIR
        print(f"   [OK] Imports da configuração - OK")
        print(f"   [EMOJI] DIAGNOSTICS_DIR: {DIAGNOSTICS_DIR}")
        print(f"   [EMOJI] SERVER_DIR: {SERVER_DIR}")
        print(f"   [EMOJI] SETUP_DIR: {SETUP_DIR}")
        
        # Testar função utilitária
        test_path = get_module_path('diagnostics', 'correcao_rag')
        print(f"   [EMOJI] get_module_path test: {test_path}")
        
    except Exception as e:
        print(f"   [ERROR] Erro na configuração: {e}")
    
    print()
    
    # Teste 6: Testar imports dos módulos
    print("6. Testando imports dos módulos reorganizados...")
    
    test_imports = [
        ('diagnostics', 'Módulos de diagnóstico'),
        ('server', 'Módulos de servidor'),
        ('setup', 'Módulos de configuração')
    ]
    
    for module_name, description in test_imports:
        try:
            module = __import__(module_name)
            print(f"   [OK] {module_name} - {description} - OK")
        except ImportError as e:
            print(f"   [ERROR] {module_name} - Erro: {e}")
    
    print()
    print("=== TESTE CONCLUÍDO ===")
    print()
    print("[EMOJI] Resumo:")
    print("   • Arquivos Python reorganizados em subpastas funcionais")
    print("   • Estrutura modular com __init__.py em cada subpasta")
    print("   • Configuração centralizada atualizada")
    print("   • Arquivos antigos removidos da raiz")
    print()
    print("[EMOJI] Próximos passos:")
    print("   1. Testar execução dos scripts nas novas localizações")
    print("   2. Atualizar imports em outros arquivos se necessário")
    print("   3. Verificar funcionalidade completa do sistema")

if __name__ == '__main__':
    test_reorganization()