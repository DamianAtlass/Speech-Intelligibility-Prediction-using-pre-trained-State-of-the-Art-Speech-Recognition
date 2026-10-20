from utils.config_dataclasses import old_get_config, old_unfold_config, Old_InferenceConfig
from utils.new_config_dataclass import InferenceConfig, load_config, convert_old_config_into_new, save_config
from pathlib import Path
from utils.logging_utils import catch_time

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import logging
logger = logging.getLogger(__name__)
from utils.evaluate_utils import get_data, evaluate_individual_run, evaluate_forced_alignment_run, get_results_dir, compare_predictors, get_inference_type
from logistic_regression import build_predictors
from utils.cuda_utils import select_device
import shutil

rows_prediction_normal = [
    ["macroscopic_entropy"],
    ["mtd"],
    ["laptev"],
    ["macroscopic_entropy", "mtd", "laptev"],

    ["entropy_kw_est"],
    ["entropy_kw_from_time_align"],
    ["tad_kw"],
    ["entropy_kw_from_time_align", "tad_kw"],
]

def evaluate_run(path: Path, device: torch.device | None = None):
    if not device:
        device = torch.device("cpu")


    config: InferenceConfig = load_config(path/"config.yaml")

    result_path = get_results_dir(path)
    if result_path.exists():
        shutil.rmtree(result_path)
    result_path.mkdir(parents=True)
    shutil.copyfile(path/"config.yaml", get_results_dir(path)/"config.yaml")

    for restrict_vocab in [False, True]:
        logger.info(f"{restrict_vocab = }")
        vocab_dir = "results_restrict_vocab" if restrict_vocab else "results_full_vocab"
        if restrict_vocab and config.model.name == "parakeet":
            continue
        with catch_time() as t:
            logger.info(f"Get data...")
            df = get_data(
                model_name=config.model.name,
                output_path=config.output_path,
                dataset_type=config.data.test_split.dataset_type,
                extract_logprobs=config.extract_logprobs,
                word_timestamps=config.word_timestamps,
                restrict_vocab=restrict_vocab,
                device=device)
        logger.info(f"Reading the generated files took: {t()/60:.2f} m")

        df["model_type"] = config.model.model_type
        # # selected_paths = df["audio_path"].unique()[:5000]
        # df_copy = df[df["snr"]<0]
        # df_copy.to_pickle(get_results_dir(config.output_path) / vocab_dir / "df_cut.pkl")
        # exit()

        if config.forced_alignment:
            evaluate_forced_alignment_run(df_forced_alignment_run=df, restrict_vocab=restrict_vocab, output_path=config.output_path)
        else:
            evaluate_individual_run(config=config, df_single_run=df, restrict_vocab=restrict_vocab)
            build_predictors(df=df, metrics_list=rows_prediction_normal, output_path=get_results_dir(config.output_path) / vocab_dir / "plots")

    compare_predictors(get_results_dir(config.output_path), get_inference_type(config))

if __name__ == '__main__':
    evaluate_run(Path("inferences/delete_me3"))