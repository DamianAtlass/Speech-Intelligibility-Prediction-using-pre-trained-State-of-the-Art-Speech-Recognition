from utils.new_config_dataclass import InferenceConfig
from pathlib import Path
import shutil

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch

from utils.evaluate_utils import get_data
from utils.dataset_utils import get_dataset, create_grid_without_bc_sentences, create_grid_bc_without_duplicates, override_noised_dataset_as_files
from utils.logging_utils import catch_time
from utils.new_config_dataclass import load_config, DataSplitConfig

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
    import nemo
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

    split_config = DataSplitConfig(dataset_type='grid', path=None, start=0, end=1, noise=False, scaling=1.0)
    dataset = get_dataset(split_config)

    model: EncDecCTCModelBPE = nemo.collections.asr.models.EncDecCTCModelBPE.restore_from(restore_path="nemo_experiments/Speech_To_Text_Finetuning/savefile.nemo")
    audio = dataset[0]["audio"]["array"]
    result = model.transcribe(audio=audio, timestamps=True)

    print(result[0].text)

def run_once():
    create_grid_bc_without_duplicates()
    create_grid_without_bc_sentences()

    grid_with_noise = "datasets/grid_with_noise"

    shutil.copytree(src="datasets/grid", dst=grid_with_noise)
    shutil.rmtree("datasets/grid_with_noise/saved_dataset")
    override_noised_dataset_as_files(target_folder=grid_with_noise)

    create_grid_without_bc_sentences(source=grid_with_noise, dest="datasets/grid_with_noise_without_bc_sentences")


if __name__ == '__main__':
    create_grid_without_bc_sentences()