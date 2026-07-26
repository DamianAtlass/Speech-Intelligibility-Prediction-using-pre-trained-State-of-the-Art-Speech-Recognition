import torch
from utils.whisper_utils import load_whisper_model
from utils.config_dataclasses import InferenceConfig
from typing import Any
from whisper import Whisper
from sip_whisper import Whisper as sip_whisper
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from utils.parakeet_utils import load_parakeet_model



def load_model(config: InferenceConfig, device: torch.device) -> Whisper|sip_whisper|EncDecCTCModelBPE:
    if config.model == "whisper":
        model: Whisper|sip_whisper = load_whisper_model(config, device)
        return model
    elif config.model=="parakeet":
        model: EncDecCTCModelBPE = load_parakeet_model(config, device)
        model.change_decoding_strategy({"strategy": "greedy_batch"})

        return model
    else:
        raise NotImplementedError