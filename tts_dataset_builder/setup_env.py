from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"[cmd] {' '.join(cmd)}")
    process = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    return process.returncode


def create_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        print(f"Criando venv em: {venv_dir}")
        venv.create(venv_dir, with_pip=True)
    else:
        print(f"Venv ja existe em: {venv_dir}")

    if os.name == "nt":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"

    if not python_exe.exists():
        raise FileNotFoundError(f"Python da venv nao encontrado: {python_exe}")

    return python_exe


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        ok = result.returncode == 0
        if ok:
            first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
            print(f"ffmpeg detectado: {first_line}")
        else:
            print("ffmpeg nao encontrado no PATH.")
        return ok
    except FileNotFoundError:
        print("ffmpeg nao encontrado no PATH.")
        return False


def download_whisper_model(python_exe: Path, project_root: Path, model_name: str) -> int:
    code = (
        "from faster_whisper import WhisperModel; "
        f"WhisperModel('{model_name}', download_root=r'{project_root / 'models'}'); "
        "print('Modelo baixado/pronto.')"
    )
    return run_cmd([str(python_exe), "-c", code], cwd=project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup automatico do ambiente Python.")
    parser.add_argument("--venv-dir", type=Path, default=Path(".venv"))
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--download-model", type=str, default="small")
    parser.add_argument("--skip-model-download", action="store_true")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    venv_dir = (project_root / args.venv_dir).resolve()
    requirements = (project_root / args.requirements).resolve()

    if not requirements.exists():
        raise FileNotFoundError(f"Arquivo requirements nao encontrado: {requirements}")

    python_exe = create_venv(venv_dir)

    if run_cmd([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], cwd=project_root) != 0:
        raise SystemExit(1)

    if run_cmd([str(python_exe), "-m", "pip", "install", "-r", str(requirements)], cwd=project_root) != 0:
        raise SystemExit(1)

    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        print("Instale ffmpeg e adicione ao PATH para suportar mais formatos de audio.")

    if not args.skip_model_download:
        if download_whisper_model(python_exe, project_root, args.download_model) != 0:
            print("Aviso: falha no download do modelo Whisper. Ele sera baixado no primeiro uso.")

    print("Setup concluido.")
    print(f"Python da venv: {python_exe}")
    if os.name == "nt":
        print(f"Ativar venv (PowerShell): {venv_dir}\\Scripts\\Activate.ps1")


if __name__ == "__main__":
    main()
