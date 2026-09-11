from utils.new_config_dataclass import InferenceConfig, load_config, convert_old_config_into_new, save_config
from pathlib import Path

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
from tqdm import tqdm

from utils.evaluate_utils import get_data, evaluate_individual_run
from utils.cuda_utils import select_device
from utils.dataset_utils import get_dataset, apply_split, get_dataset_dict, apply_filter
from utils.logging_utils import catch_time
from utils.new_config_dataclass import load_config, DatasetConfig, DataSplitConfig

def inspect_df(path: Path, device: torch.device | None = None):
    if not device:
        device = torch.device("cpu")


    config: InferenceConfig = load_config(path/"config.yaml")

    with catch_time() as t:
        df = get_data(
            config.model.name,
            config.output_path,
            config.data.test_split.dataset_type,
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
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

    device = select_device()
    model: EncDecCTCModelBPE = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="nvidia/parakeet-ctc-0.6b").to(device)
    model.change_decoding_strategy({"decoding_cfg": "greedy_batch}"})

    dataset = get_dataset("grid")
    dataset = apply_split(dataset, test_split=100, train_split=0, val_split=0)
    dataset = dataset["test"]

    with catch_time() as t:
        transcriptions = model.transcribe(
            audio=[sample["audio"]["array"] for sample in dataset],
            timestamps=True,
            batch_size=20
        )
    print(f"Execution time of do_something: {t():.1f} s")

    result = transcriptions[0]
    print()

def create_grid_without_bc_sentences():
    config = DatasetConfig(
        train_split=DataSplitConfig(dataset_type='grid', path=None, start=0, end=1.0, noise=False, scaling=1.0),
        val_split=DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=1.0, noise=False, scaling=1.0),)
    dataset_dict = get_dataset_dict(config)

    sentences_in_grid = dataset_dict["train"].unique("sentence")
    sentences_in_grid_bc = dataset_dict["test"].unique("sentence")
    sentences_in_not_in_bc = set(sentences_in_grid) - set(sentences_in_grid_bc)

    dataset_dict["train"] = apply_filter(dataset_dict["train"], {"sentence": sentences_in_not_in_bc})

    save_at = Path.cwd() / "datasets/grid_without_bc_sentences" / "saved_dataset"
    save_at.mkdir(parents=True, exist_ok=True)
    dataset_dict["train"].save_to_disk(save_at)

def create_grid_bc_without_duplicates():
    split_config = DataSplitConfig(dataset_type='grid_bc', path=None, start=0, end=1.0, noise=False, scaling=1.0)
    dataset = get_dataset(split_config)

    seen = set()
    indices = []

    for i, sentence in tqdm(enumerate(dataset["sentence"])):
        if sentence not in seen:
            seen.add(sentence)
            indices.append(i)

    dataset = dataset.select(indices)

    from collections import Counter
    print(Counter(dataset["snr_db"]))

    save_at = Path.cwd() / "datasets/grid_bc_without_duplicates" / "saved_dataset"
    save_at.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_at)

if __name__ == '__main__':
    create_grid_without_bc_sentences()
    create_grid_bc_without_duplicates()