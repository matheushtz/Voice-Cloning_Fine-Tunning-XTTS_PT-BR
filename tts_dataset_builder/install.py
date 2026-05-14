"""
Script de instalacao automatica das dependencias para XTTS v2 fine-tuning.
Detecta CUDA automaticamente e instala pacotes compatibles.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> int:
    """Executa comando shell."""
    print(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def upgrade_pip() -> None:
    """Atualiza pip, setuptools e wheel."""
    print("\n=== Atualizando pip, setuptools, wheel ===")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])


def install_torch_cuda() -> bool:
    """
    Detecta CUDA e instala PyTorch com CUDA se disponível.
    Retorna True se CUDA foi detectado, False caso contrário.
    """
    print("\n=== Verificando CUDA ===")

    # Tenta importar torch se já estiver instalado para verificar CUDA
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA detectado! Versao: {torch.version.cuda}")
            print(f"GPU disponivel: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("CUDA nao detectado, usando CPU")
            return False
    except ImportError:
        pass

    # Instala PyTorch com CUDA
    print("Instalando PyTorch com CUDA support...")
    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchaudio",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
    )

    # Verifica novamente
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA ativado com sucesso! Versao: {torch.version.cuda}")
            return True
    except ImportError:
        pass

    print("Usando CPU (CUDA nao disponivel)")
    return False


def install_requirements(requirements_file: Path) -> None:
    """Instala dependencias de requirements.txt."""
    if not requirements_file.exists():
        raise FileNotFoundError(f"requirements.txt nao encontrado: {requirements_file}")

    print(f"\n=== Instalando requirements de {requirements_file} ===")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])


def enforce_tts_compatibility() -> None:
    """Garante versao de transformers compativel com TTS 0.22.0."""
    print("\n=== Ajustando compatibilidade do TTS (transformers==4.39.3) ===")
    run_cmd([sys.executable, "-m", "pip", "install", "transformers==4.39.3"])


def verify_installation() -> None:
    """Verifica se pacotes principais estao instalados."""
    print("\n=== Verificando instalacao ===")

    packages = {
        "torch": "PyTorch",
        "torchaudio": "torchaudio",
        "TTS": "Coqui TTS",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "numpy": "numpy",
        "pandas": "pandas",
    }

    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (NAO INSTALADO)")
            all_ok = False

    if all_ok:
        print("\nTodas as dependencias foram instaladas com sucesso!")
    else:
        print("\nAlgumas dependencias nao foram instaladas.")
        sys.exit(1)

    # Mostra versoes e CUDA status
    try:
        import torch

        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Instalar dependencias para XTTS v2 fine-tuning")
    parser.add_argument("--no-upgrade", action="store_true", help="Pular upgrade de pip")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    requirements_file = project_root / "requirements.txt"

    try:
        if not args.no_upgrade:
            upgrade_pip()

        install_torch_cuda()
        install_requirements(requirements_file)
        enforce_tts_compatibility()
        verify_installation()

        print("\n" + "=" * 60)
        print("Instalacao concluida com sucesso!")
        print("=" * 60)

    except Exception as e:
        print(f"\nErro durante instalacao: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
