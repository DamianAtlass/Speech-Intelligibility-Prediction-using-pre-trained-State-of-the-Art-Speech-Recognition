import torch
from datasets import DatasetDict
from speech_to_text_finetune import train_parakeet

from utils.new_config_dataclass import TrainingConfig
from train_whisper import train_whisper


def training(config: TrainingConfig, dataset: DatasetDict, device: torch.device):
    match config.model.name:
        case "whisper":
            train_whisper(config, dataset, device)
        case "parakeet":
            train_parakeet(config, dataset, device)
        case _:
            raise NotImplementedError