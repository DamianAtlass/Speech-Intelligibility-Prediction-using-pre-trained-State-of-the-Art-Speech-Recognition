import nemo.collections.asr as nemo_asr
import numpy
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from utils.new_config_dataclass import InferenceConfig
import numpy as np
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch

from utils.cuda_utils import select_device
from utils.dataset_utils import get_dataset, apply_split
from utils.logging_utils import catch_time
from math import ceil
from typing import Callable



def load_parakeet_model(config: InferenceConfig, device: torch.device):
    model: EncDecCTCModelBPE = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=f"nvidia/{config.model.name}-{config.model.model_type}").to(device)
    return model

def get_collate_fn(device: torch.device) -> Callable:

    def collate(batch):
        lengths = np.array([b["audio"]["array"].shape[-1] for b in batch], dtype=np.int64)
        max_len = lengths.max()

        audio = np.stack([np.pad(b["audio"]["array"],(0, max_len - b["audio"]["array"].shape[-1]),)
            for b in batch
        ])

        return (
            torch.from_numpy(audio).float().to(device),
            torch.from_numpy(lengths).to(device),
        )
    return collate

def main():
    print("sdlfuj")
    device = select_device()
    model: EncDecCTCModelBPE = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="nvidia/parakeet-ctc-0.6b").to(device)

    model.change_decoding_strategy({"decoding_cfg": "greedy_batch}"})

    dataset = get_dataset("grid")
    num_samples = 110
    dataset = apply_split(dataset, val_split=num_samples, train_split=0, test_split=0)
    dataset = dataset["val"]
    batch_size = 20

    for i in range(ceil(num_samples / batch_size)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(dataset))
        subset = dataset.select(range(start,end))

        dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate)
        with catch_time() as t:
            transcriptions = model.transcribe(
                audio=dataloader,
                timestamps=True,
            )
        print(f"Execution time of do_something: {t():.1f} s")

        result = transcriptions[0]
        print()

if __name__ == '__main__':
    main()


