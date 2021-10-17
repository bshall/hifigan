from pathlib import Path
import numpy as np
import argparse
import torch
import torchaudio
from tqdm import tqdm


def generate(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading checkpoint")
    hifigan = torch.hub.load("bshall/hifigan:main", args.model_name).cuda()

    print(f"Generating audio from {args.in_dir}")
    for path in tqdm(list(args.in_dir.rglob("*.npy"))):
        mel = torch.from_numpy(np.load(path))
        mel = mel.unsqueeze(0).cuda()

        wav, sr = hifigan.generate(mel)
        wav = wav.squeeze(0).cpu()

        out_path = args.out_dir / path.relative_to(args.in_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        torchaudio.save(out_path.with_suffix(".wav"), wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio for a directory of mel-spectrogams using HiFi-GAN."
    )
    parser.add_argument(
        "in-dir",
        help="path to directory containing the mel-spectrograms",
        type=Path,
    )
    parser.add_argument(
        "out-dir",
        help="path to output directory",
        type=Path,
    )
    parser.add_argument(
        "--model-name",
        help="available models",
        choices=["hifigan", "hifigan-hubert-soft", "hifigan-hubert-discrete"],
        default="hifigan-hubert-soft",
    )
    args = parser.parse_args()

    generate(args)
