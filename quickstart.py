"""
Wrapper para executar scripts do tts_dataset_builder a partir da raiz do projeto.
"""

import subprocess
import sys
from pathlib import Path

# Localiza a pasta do projeto
project_root = Path(__file__).resolve().parent
tts_builder_dir = project_root / "tts_dataset_builder"

if not tts_builder_dir.exists():
    print(f"Erro: tts_dataset_builder nao encontrado em {project_root}")
    sys.exit(1)

# Executa quickstart.py dentro de tts_dataset_builder
script_path = tts_builder_dir / "quickstart.py"
if not script_path.exists():
    print(f"Erro: {script_path} nao encontrado")
    sys.exit(1)

print(f"Executando: {script_path}\n")
result = subprocess.run([sys.executable, str(script_path)], cwd=str(tts_builder_dir))
sys.exit(result.returncode)
