from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt


def main() -> None:
    train_gpt(
        language="pt",
        num_epochs=10,
        batch_size=2,
        grad_acumm=8,
        train_csv=r"D:\User\Desktop\TTS-VOICE-TRAINING\dataset\metadata_f.csv",
        eval_csv=r"D:\User\Desktop\TTS-VOICE-TRAINING\dataset\metadata_f.csv",
        output_path=r"D:\User\Desktop\TTS-VOICE-TRAINING\tts_dataset_builder\xtts_training",
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
