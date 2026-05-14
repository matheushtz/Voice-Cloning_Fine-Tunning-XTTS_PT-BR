"""
Detecta automaticamente os paths do projeto independente de onde é executado.
"""

from pathlib import Path

def get_project_paths():
    """
    Retorna dicionario com paths do projeto, automaticamente detectado.
    Funciona se executado da raiz ou de dentro de tts_dataset_builder.
    """
    
    # Identifica onde estamos
    current_file = Path(__file__).resolve()
    
    # Se estamos em tts_dataset_builder, volta para raiz
    if current_file.parent.name == "tts_dataset_builder":
        project_root = current_file.parent.parent
        tts_dir = current_file.parent
    else:
        # Estamos na raiz
        project_root = current_file.parent
        tts_dir = project_root / "tts_dataset_builder"
    
    return {
        "project_root": project_root,
        "tts_builder_dir": tts_dir,
        "dataset_dir": project_root / "dataset",
        "output_dir": tts_dir / "output",
        "checkpoints_dir": tts_dir / "checkpoints",
        "models_dir": tts_dir / "models",
    }

if __name__ == "__main__":
    paths = get_project_paths()
    for key, value in paths.items():
        print(f"{key}: {value}")
