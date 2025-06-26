#!/usr/bin/env python3
"""
Script para instalar PyTorch com suporte CUDA
"""

import subprocess
import sys

def run_command(cmd):
    """Executa comando e retorna resultado"""
    print(f"Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0

def main():
    print("=== Instalação do PyTorch com CUDA ===")
    
    # 1. Desinstalar PyTorch CPU atual
    print("\n1. Desinstalando PyTorch CPU...")
    if not run_command([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"]):
        print("Erro ao desinstalar PyTorch")
        return False
    
    # 2. Instalar PyTorch com CUDA 12.1 (compatível com CUDA 12.9)
    print("\n2. Instalando PyTorch com CUDA 12.1...")
    pytorch_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    if not run_command(pytorch_cmd):
        print("Erro ao instalar PyTorch com CUDA")
        return False
    
    # 3. Verificar instalação
    print("\n3. Verificando instalação...")
    verify_cmd = [
        sys.executable, "-c", 
        "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    ]
    
    if not run_command(verify_cmd):
        print("Erro ao verificar instalação")
        return False
    
    print("\n✅ PyTorch com CUDA instalado com sucesso!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)