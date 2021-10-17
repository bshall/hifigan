# HiFi-GAN

An 16kHz implementation of HiFi-GAN for [soft-vc](https://github.com/bshall/soft-vc).

Relevant links:
- [Official HiFi-GAN repo](https://github.com/jik876/hifi-gan)
- [HiFi-GAN paper](https://arxiv.org/abs/2010.05646)
- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper]()

## Example Usage

```python
import torch
import numpy as np

# Load checkpoint
hifigan = torch.hub.load("bshall/hifigan:main", "hifigan-hubert-soft").cuda()
# Load mel-spectrogram
mel = torch.from_numpy(np.load("path/to/mel")).unsqueeze(0).cuda()
# Generate
wav, sr = hifigan.generate(mel)
```

## Train

**Step 1**: Download and extract the [LJ-Speech dataset](https://keithito.com/LJ-Speech-Dataset/)

**Step 2**: Resample the audio to 16kHz:
```
usage: resample.py [-h] [--sample-rate SAMPLE_RATE] in-dir out-dir

Resample an audio dataset.

positional arguments:
  in-dir                path to the dataset directory
  out-dir               path to the output directory

optional arguments:
  -h, --help            show this help message and exit
  --sample-rate SAMPLE_RATE
                        target sample rate (default 16kHz)
```

**Step 3**: Download the dataset splits and move them into the root of the dataset directory.
After steps 2 and 3 your dataset directory should look like this:
```
LJSpeech-1.1
│   test.txt
│   train.txt
│   validation.txt
├───mels
└───wavs
```
Note: the mels directory is optional. If you want to fine-tune HiFi-GAN the mels directory should contain ground-truth aligned spectrograms from an acoustic model.

**Step 4**: Train HiFi-GAN:
```
usage: train.py [-h] [--resume RESUME] [--finetune] dataset-dir checkpoint-dir

Train or finetune HiFi-GAN.

positional arguments:
  dataset-dir      path to the preprocessed data directory
  checkpoint-dir   path to the checkpoint directory

optional arguments:
  -h, --help       show this help message and exit
  --resume RESUME  path to the checkpoint to resume from
  --finetune       whether to finetune (note that a resume path must be given)
```

## Generate
To generate using the trained HiFi-GAN models, see [Example Usage](#example-usage) or use the `generate.py` script:

```
usage: generate.py [-h] [--model-name {hifigan,hifigan-hubert-soft,hifigan-hubert-discrete}] in-dir out-dir

Generate audio for a directory of mel-spectrogams using HiFi-GAN.

positional arguments:
  in-dir                path to directory containing the mel-spectrograms
  out-dir               path to output directory

optional arguments:
  -h, --help            show this help message and exit
  --model-name {hifigan,hifigan-hubert-soft,hifigan-hubert-discrete}
                        available models
```

## Acknowledgements
This repo is based heavily on [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan).