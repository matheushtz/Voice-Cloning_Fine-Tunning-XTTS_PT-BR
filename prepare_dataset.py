"""Wrapper para prepare_dataset.py"""
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
script_path = project_root / "tts_dataset_builder" / "prepare_dataset.py"

if not script_path.exists():
    print(f"Erro: {script_path} nao encontrado")
    sys.exit(1)

result = subprocess.run([sys.executable, str(script_path)], cwd=str(project_root / "tts_dataset_builder"))
sys.exit(result.returncode)
