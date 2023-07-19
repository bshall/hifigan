dependencies = ["torch", "torchaudio"]

URLS = {
    "hifigan": "https://github.com/bshall/hifigan/releases/download/v0.1/hifigan-67926ec6.pt",
    "hifigan-hubert-discrete": "https://github.com/bshall/hifigan/releases/download/v0.1/hifigan-hubert-discrete-bbad3043.pt",
    "hifigan-hubert-soft": "https://github.com/bshall/hifigan/releases/download/v0.1/hifigan-hubert-soft-65f03469.pt",
}

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from hifigan.generator import HifiganGenerator


def _hifigan(
    name: str,
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> HifiganGenerator:
    hifigan = HifiganGenerator()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[name], map_location=map_location, progress=progress
        )
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        hifigan.load_state_dict(checkpoint)
        hifigan.eval()
        hifigan.remove_weight_norm()
    return hifigan


def hifigan(
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> HifiganGenerator:
    """HiFiGAN Vocoder from from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        map_location: a function or a dict specifying how to remap storage locations (see torch.load)
    """
    return _hifigan("hifigan", pretrained, progress, map_location)


def hifigan_hubert_soft(
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> HifiganGenerator:
    """HiFiGAN Vocoder from from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Finetuned on spectrograms generated from the soft acoustic model.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        map_location: a function or a dict specifying how to remap storage locations (see torch.load)
    """
    return _hifigan(
        "hifigan-hubert-soft", pretrained, progress, map_location=map_location
    )


def hifigan_hubert_discrete(
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> HifiganGenerator:
    """HiFiGAN Vocoder from from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Finetuned on spectrograms generated from the discrete acoustic model.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        map_location: a function or a dict specifying how to remap storage locations (see torch.load)
    """
    return _hifigan(
        "hifigan-hubert-discrete", pretrained, progress, map_location=map_location
    )
