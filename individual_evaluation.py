from pathlib import Path
from utils.config_dataclasses import get_config
from utils.evaluate_utils import get_data, plot_wer_to_snr, plot_needleman_wunsch_wer_to_snr
import torch
import pandas as pd
import json

from utils.logging_utils import catch_time


def foo():
    indi_eval_folder = Path.cwd() / "individual_evaluations"

    subfolder_name = indi_eval_folder / "tmp"
    if not subfolder_name.exists():
        subfolder_name.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    runs = [
        {"name": "small (trained for 1 epoch)",
         "path": Path.cwd() / "inferences/small_trained"},

        {"name": "small (trained for 2 epochs)",
         "path": Path.cwd() / "inferences/small_trained2"},

        {"name": "small",
         "path": Path.cwd() / "inferences/small_default"},
    ]

    with open(subfolder_name/"models.json", 'w') as f:
        runs_tmp = runs.copy()
        for r in runs_tmp:
            r["path"] = str(r["path"].relative_to(Path.cwd()))
        json.dump(runs, f, indent=4)

    # step 1: collect data
    df = None
    for run in runs:
        config = get_config(Path(run["path"]) / "config.ini")
        with catch_time() as t:
            summary, df_single_run = get_data(config.output_path, config.dataset_type, device)
        print(f"Reading the generated files took: {t():.4f} s")
        df_single_run["model_type"] = config.model_type
        df_single_run["name"] = run["name"]

        if df is None:
            df = df_single_run
        else:
            df = pd.concat([df, df_single_run], ignore_index=True)

    #step 2: plot

    plot_wer_to_snr(df,
                    shifting_attribute="name",
                    shifting_attribute_label="different models",
                    output_path=subfolder_name)

    plot_needleman_wunsch_wer_to_snr(df,
                    shifting_attribute="name",
                    shifting_attribute_label="different models",
                    output_path=subfolder_name)


if __name__ == '__main__':
    foo()