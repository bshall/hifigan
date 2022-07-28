from pathlib import Path
import numpy as np
import argparse
import torch
import torchaudio
from tqdm import tqdm


def generate(args):
    print("Loading checkpoint")
    model_name = f"hifigan_hubert_{args.model}" if args.model != "base" else "hifigan"
    hifigan = torch.hub.load("bshall/hifigan:main", model_name).cuda()

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
        "model",
        help="available models (HuBERT-Soft, HuBERT-Discrete, or Base).",
        choices=["soft", "discrete", "base"],
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to input directory containing the mel-spectrograms.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to output directory.",
        type=Path,
    )
    args = parser.parse_args()

    generate(args)
