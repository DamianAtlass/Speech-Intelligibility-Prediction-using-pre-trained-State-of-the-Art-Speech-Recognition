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
from utils.cuda_utils import select_device


def evaluate_run(path: Path, device: torch.device | None = None):
    if not device:
        device = torch.device("cpu")


    config = load_config(path/"config.yaml")


    with catch_time() as t:
        df_single_run = get_data(
            config.model.name,
            config.output_path,
            config.data.val_split.dataset_type,
            config.extract_logprobs,
            config.word_timestamps,
            device)
    print(f"Reading the generated files took: {t():.4f} s")

    df_single_run["model_type"] = config.model.model_type
    evaluate_individual_run(config=config, df_single_run=df_single_run, device=device)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", 10000):
        #print(df_single_run[["references_kw", "machine_transcripts", "machine_transcripts_kw", "wers_machine_kw"]])
if __name__ == '__main__':
    evaluate_run(Path("inferences/delete_me3"))