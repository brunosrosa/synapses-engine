#!/usr/bin/env python3
"""
Script para verificar configuração CUDA e PyTorch
"""

import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Current device: CPU only")
except ImportError as e:
    print(f"Erro ao importar PyTorch: {e}")
    sys.exit(1)

# Verificar se NVIDIA drivers estão instalados
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n=== NVIDIA-SMI Output ===")
        print(result.stdout)
    else:
        print("\nNVIDIA-SMI não encontrado ou erro ao executar")
except Exception as e:
    print(f"\nErro ao verificar nvidia-smi: {e}")

# Verificar versão do CUDA toolkit instalada
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n=== NVCC Version ===")
        print(result.stdout)
    else:
        print("\nNVCC não encontrado - CUDA toolkit pode não estar instalado")
except Exception as e:
    print(f"\nErro ao verificar nvcc: {e}")