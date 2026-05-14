"""
Quick start script para pipeline completo XTTS v2.
Execute em sequencia para setup -> preparacao -> treino -> inferencia.
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_script(script_name: str, description: str) -> bool:
    """Executa um script Python."""
    print("\n" + "=" * 70)
    print(f"ETAPA: {description}")
    print("=" * 70)
    
    script_path = Path(__file__).resolve().parent / script_name
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=False)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Erro ao executar {script_name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick start: pipeline completo XTTS v2 fine-tuning"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        help="Executar apenas uma etapa (1=install, 2=prepare, 3=train, 4=inference)",
    )
    parser.add_argument("--skip-install", action="store_true", help="Pular instalacao")
    args = parser.parse_args()

    steps = [
        (1, "install.py", "1. Instalacao de dependencias"),
        (2, "prepare_dataset.py", "2. Validacao e preparacao do dataset"),
        (3, "train_xtts.py", "3. Fine-tuning do modelo XTTS v2"),
        (4, "inference.py", "4. Geracao de audio sintetizado"),
    ]

    print("=" * 70)
    print("XTTS v2 FINE-TUNING - QUICK START")
    print("=" * 70)
    print("\nEste script executa o pipeline completo:")
    print("  1. Instala dependencias (CUDA auto-detect)")
    print("  2. Prepara e valida dataset")
    print("  3. Treina modelo com sua voz")
    print("  4. Gera sintese de fala\n")

    if args.skip_install:
        steps = steps[1:]

    if args.step:
        steps = [s for s in steps if s[0] == args.step]

    success_count = 0
    for step_num, script_name, description in steps:
        if run_script(script_name, description):
            success_count += 1
            logger.info(f"✓ {description} - OK")
        else:
            logger.error(f"✗ {description} - FALHOU")
            logger.error(f"Verifique os logs acima e rode novamente.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"SUCESSO! {success_count}/{len(steps)} etapas completadas")
    print("=" * 70)
    print(f"\nOutput final: output/output.wav")
    print("Seu modelo foi treinado e testado com sucesso!")
    print("\nPara usar outro texto, edite inference.py e rode novamente.")


if __name__ == "__main__":
    main()
