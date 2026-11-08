import shutil
from pathlib import Path
from utils.paths import PROJECT_ROOT
from utils.new_config_dataclass import load_config
from utils.evaluate_utils import get_data, PredictionManager, get_results_dir, get_inference_type, CorrelationManager, corr_or_mut, evaluate_transcripts
from utils.plotting_utils import plot_wer_to_snr, boxplot_corr_per_listener, plot_x_to_snr
import torch
import pandas as pd
import json


from utils.logging_utils import catch_time


def compare_graphs():
    indi_eval_folder = Path.cwd() / "individual_evaluations"

    subfolder_name = indi_eval_folder / "default_performance"
    if not subfolder_name.exists():
        subfolder_name.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')

    runs = [
        {"name": "Whisper (turbo)",
         "path": Path.cwd() / "inferences/turbo_default_bc2",
         "restricted_vocab": False,},
        {"name": "Parakeet (ctc-1.1b)",
         "path": Path.cwd() / "inferences/parakeet_1.1_default",
         "restricted_vocab": False,},
    ]

    with open(subfolder_name/"models.json", 'w') as f:
        runs_tmp = runs.copy()
        for r in runs_tmp:
            r["path"] = str(r["path"].relative_to(Path.cwd()))
        json.dump(runs, f, indent=4)

    # step 1: collect data
    df = None
    for run in runs:
        config = load_config((PROJECT_ROOT/run["path"]) / "config.yaml")
        with catch_time() as t:
            df_single_run = get_data(
                config.model.name,
                config.output_path,
                config.data.test_split.dataset_type,
                config.extract_logprobs,
                config.word_timestamps,
                run["restricted_vocab"],
                device)
        print(evaluate_transcripts(df_single_run))
        print(f"Reading the generated files took: {t():.4f} s")
        df_single_run["model_type"] = config.model.model_type
        df_single_run["name"] = run["name"]

        if df is None:
            df = df_single_run
        else:
            df = pd.concat([df, df_single_run], ignore_index=True)

    #step 2: plot
    plot_wer_to_snr(
        df[["human_transcript_kw", "machine_transcript_kw", "snr", "reference", "reference_kw", "name"]],
        trans_col="machine_transcript_kw",
        ref_col="reference_kw",
        shifting_attribute="name",
        output_path=subfolder_name)

    boxplot_corr_per_listener(
        df[["wer_human_kw", "wer_machine_kw", "name", "listener"]],
        col_name="wer_machine_kw",
        model="whisper",
        model_type="",
        shifting_attribute="name",
        output_path=subfolder_name)

    if False:
        plot_x_to_snr(df=df,
                      plotting_attribute="macroscopic_entropy",
                      shifting_attribute_label="model",
                      shifting_attribute="name",
                      output_path=subfolder_name
                      )

def generate_predictor_summary():
    paths = [
        "inferences/delete_me_parakeet/",
    ]
    configs = [load_config(Path(p)/"config.yaml") for p in paths]
    PredictionManager.compare_predictors(configs)

def generate_correlation_summary():
    paths = [
        "inferences/delete_me/",
    ]
    configs = [load_config(Path(p) / "config.yaml") for p in paths]
    CorrelationManager.compare_correlations(configs)

if __name__ == '__main__':
    compare_graphs()