from utils.config_dataclasses import old_get_config, old_unfold_config, Old_InferenceConfig
from utils.new_config_dataclass import InferenceConfig, load_config, convert_old_config_into_new, save_config
from pathlib import Path
from utils.logging_utils import catch_time

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import logging
logger = logging.getLogger(__name__)
from utils.evaluate_utils import get_data, evaluate_individual_run

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
from utils.cuda_utils import select_device
from utils.dataset_utils import get_dataset, apply_split
from utils.logging_utils import catch_time

def inspect_df(path: Path, device: torch.device | None = None):
    if not device:
        device = torch.device("cpu")


    config: InferenceConfig = load_config(path/"config.yaml")

    with catch_time() as t:
        df = get_data(
            config.model.name,
            config.output_path,
            config.data.val_split.dataset_type,
            config.extract_logprobs,
            config.word_timestamps,
            device)
    print(f"Reading the generated files took: {t():.4f} s")

    print()

    group = df.groupby("listener")
    for g in group:
        print(f"listener: {g[0]}")

        print("samples: ", len(g[1]))

def nemo_sandbox():
    device = select_device()
    model: EncDecCTCModelBPE = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="nvidia/parakeet-ctc-0.6b").to(device)
    model.change_decoding_strategy({"decoding_cfg": "greedy_batch}"})

    dataset = get_dataset("grid")
    dataset = apply_split(dataset, val_split=100, train_split=0, test_split=0)
    dataset = dataset["val"]

    with catch_time() as t:
        transcriptions = model.transcribe(
            audio=[sample["audio"]["array"] for sample in dataset],
            timestamps=True,
            batch_size=20
        )
    print(f"Execution time of do_something: {t():.1f} s")

    result = transcriptions[0]
    print()

if __name__ == '__main__':
    inspect_df(Path("inferences/turbo_exp2_bc"))