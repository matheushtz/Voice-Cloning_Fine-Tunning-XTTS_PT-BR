"""Wrapper para train_xtts.py"""
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    script_path = project_root / "tts_dataset_builder" / "train_xtts.py"

    if not script_path.exists():
        print(f"Erro: {script_path} nao encontrado")
        return 1

    result = subprocess.run([sys.executable, str(script_path)], cwd=str(project_root / "tts_dataset_builder"))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
