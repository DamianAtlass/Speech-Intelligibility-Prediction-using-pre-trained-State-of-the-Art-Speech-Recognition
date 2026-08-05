import shutil
from pathlib import Path
from utils.new_config_dataclass import load_config
from utils.evaluate_utils import get_data
from utils.plotting_utils import plot_wer_to_snr, boxplot_corr_per_listener, plot_x_to_snr
import torch
import pandas as pd
import json

from utils.logging_utils import catch_time


def foo():
    indi_eval_folder = Path.cwd() / "individual_evaluations"

    subfolder_name = indi_eval_folder / "compare_parakeet"
    if not subfolder_name.exists():
        subfolder_name.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    runs = [
        {"name": "parakeet (ctc-0.6b) untrained",
         "path": Path.cwd() / "inferences/ctc-0.6b"},
        {"name": "parakeet (ctc-1.1b) untrained",
         "path": Path.cwd() / "inferences/ctc-1.1b"},
        {"name": "whisper large-v3-turbo untrained",
         "path": Path.cwd() / "inferences/turbo_default_bc"},
        {"name": "tdt_ctc-1.1b untrained",
         "path": Path.cwd() / "inferences/tdt_ctc-1.1b"},
    ]

    with open(subfolder_name/"models.json", 'w') as f:
        runs_tmp = runs.copy()
        for r in runs_tmp:
            r["path"] = str(r["path"].relative_to(Path.cwd()))
        json.dump(runs, f, indent=4)

    # step 1: collect data
    df = None
    for run in runs:
        config = load_config(Path(run["path"]) / "config.yaml")
        with catch_time() as t:
            df_single_run = get_data(config.model.name, config.output_path, config.data.val_split.dataset_type, config.extract_logprobs, device)
        print(f"Reading the generated files took: {t():.4f} s")
        df_single_run["model_type"] = config.model.model_type
        df_single_run["name"] = run["name"]

        if df is None:
            df = df_single_run
        else:
            df = pd.concat([df, df_single_run], ignore_index=True)

    #step 2: plot
    plot_wer_to_snr(
        df[["human_transcripts_kw", "machine_transcripts", "snr", "references", "references_kw", "name"]],
        only_kw=False,
        shifting_attribute="name",
        shifting_attribute_label="\ndifferent models",
        output_path=subfolder_name)

    boxplot_corr_per_listener(
        df[["wers_human_kw", "wers_machine_kw", "name", "listener"]],
        correlate_to="wers_machine_kw",
        model="whisper",
        model_type="",
        shifting_attribute="name",
        output_path=subfolder_name)

    if False:
        plot_x_to_snr(df=df,
                      plotting_attribute="average_macroscopic_entropy",
                      shifting_attribute_label="model",
                      shifting_attribute="name",
                      output_path=subfolder_name
                      )


if __name__ == '__main__':
    foo()