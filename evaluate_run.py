from utils.config_dataclasses import old_get_config, old_unfold_config, Old_InferenceConfig
from utils.new_config_dataclass import InferenceConfig, load_config, convert_old_config_into_new, save_config
from pathlib import Path
from utils.logging_utils import catch_time

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import logging
logger = logging.getLogger(__name__)
from utils.evaluate_utils import get_data, evaluate_individual_run, evaluate_forced_alignment_run


def evaluate_run(path: Path, device: torch.device | None = None):
    if not device:
        device = torch.device("cpu")


    config: InferenceConfig = load_config(path/"config.yaml")
    for restrict_vocab in [False]:
        with catch_time() as t:
            df = get_data(
                model_name=config.model.name,
                output_path=config.output_path,
                dataset_type=config.data.test_split.dataset_type,
                extract_logprobs=config.extract_logprobs,
                word_timestamps=config.word_timestamps,
                restrict_vocab=restrict_vocab,
                device=device)
        print(f"Reading the generated files took: {t():.4f} s")

        df["model_type"] = config.model.model_type
        if config.forced_alignment:
            evaluate_forced_alignment_run(config, df, restrict_vocab)
        else:
            evaluate_individual_run(config=config, df_single_run=df, restrict_vocab=restrict_vocab)

if __name__ == '__main__':
    evaluate_run(Path("inferences/delete_me3"))