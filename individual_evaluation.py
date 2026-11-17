import shutil
from pathlib import Path
from utils.paths import PROJECT_ROOT
from utils.new_config_dataclass import load_config
from utils.evaluate_utils import get_data, PredictionManager, get_results_dir, get_inference_type, CorrelationManager, corr_or_mut, evaluate_transcripts
from utils.plotting_utils import plot_wer_to_snr, boxplot_corr_per_listener, plot_x_to_snr, barplot_finetuning_progress, heatmap
import torch
import pandas as pd
import json

from utils.logging_utils import catch_time

def merge_dataset_from_multiple_runs(runs: list[dict]):
    device = torch.device('cpu')

    # step 1: collect data
    df: pd.DataFrame = None
    for run in runs:
        config = load_config((PROJECT_ROOT / run["path"]) / "config.yaml")
        with catch_time() as t:
            print(f"get dataset at {run["path"]}")
            df_single_run = get_data(
                config.model.name,
                config.output_path,
                config.data.test_split.dataset_type,
                config.extract_logprobs,
                config.word_timestamps,
                run["restricted_vocab"],
                device)
        #print(evaluate_transcripts(df_single_run))
        print(f"Reading the generated files took: {t():.1f} s")
        df_single_run["model_type"] = config.model.model_type
        df_single_run["name"] = run["name"]

        if df is None:
            df = df_single_run
        else:
            df = pd.concat([df, df_single_run], ignore_index=True)

    return df

class Evaluator:

    def __init__(self, runs: list[dict], save_at: str):
        self.runs = runs
        self.df = merge_dataset_from_multiple_runs(runs)
        self.save_at = indi_eval_folder / save_at

        if not self.save_at.exists():
            self.save_at.mkdir(parents=True, exist_ok=True)

        with open(self.save_at / "models.json", 'w') as f:
            json.dump(self.runs, f, indent=4)

    def plot_wer(self):
        plot_wer_to_snr(
            self.df[["human_transcript_kw", "machine_transcript_kw", "snr", "reference", "reference_kw", "name"]],
            trans_col="machine_transcript_kw",
            ref_col="reference_kw",
            shifting_attribute="name",
            output_path=self.save_at)

    def plot_corr(self):
        boxplot_corr_per_listener(
            df=self.df[["wer_human_kw", "wer_machine_kw", "name", "listener"]],
            col_name="wer_machine_kw",
            model="whisper",
            model_type="",
            shifting_attribute="name",
            output_path=self.save_at)

    def plot_training_progress(self):
        barplot_finetuning_progress(
            df=self.df[["reference_kw", "machine_transcript_kw_est", "name", "snr"]],
            col_label="foo",
            shifting_attribute="name",
            output_path=self.save_at)

    def calculate_stoi_corr(self):
        with open("stoi_data/grid_bc_stoi.json") as f:
            stoi_data: dict = json.load(f)

        clean_grid_audios = [stoi_data[path] for path in self.df["audio_path"]]

        self.df["stoi_score"] = clean_grid_audios

        c = corr_or_mut(df=self.df, wer_col_or_kw_col="wer_human", col_name="stoi_score",
                    special_metric="spearman_correlation", kw_idx=None)

        print(c)
    def foo(self, metric_column: str):
        def kw_recognized(x):
            return [t == r for t,r in zip(x["kw_tmp"], x["human_transcript_kw"])]
        kw_column = "machine_transcript_kw_est" if "kw_est" in metric_column else "machine_transcript_kw_from_time_align"

        self.df["kw_tmp"] = self.df[kw_column]
        self.df["correctly_recognized"] = self.df[["kw_tmp", "human_transcript_kw"]].apply(kw_recognized, axis=1)
        self.df.drop(("kw_tmp"), axis=1)

        self.df["correctly_recognized"] = self.df["correctly_recognized"].apply(lambda x: x[0])
        self.df[metric_column] = self.df[metric_column].apply(lambda x: x[0])

        heatmap(self.df, col_label=metric_column)





indi_eval_folder = Path.cwd() / "individual_evaluations"


def compare_graphs():
    if False:
        runs = [
            {"name": "parakeet default",
             "path": "inferences/delete_me_parakeet",
             "restricted_vocab": False,},
            {"name": "parakeet fine-tuned",
             "path": "inferences/grid_bc_ctc-1.1b_v4",
             "restricted_vocab": False, },
        ]
        e = Evaluator(runs, save_at="wip")
        e.plot_wer()

        e.plot_training_progress()

        runs = [
            {"name": "whisper default",
             "path": "inferences/dummy_bc_base_default",
             "restricted_vocab": False, },
            {"name": "whisper fine-tuned",
             "path": "inferences/grid_bc_turbo_v4",
             "restricted_vocab": False, },
        ]
        e = Evaluator(runs, save_at="wip")
        e.plot_wer()

        e.plot_training_progress()

        runs = [
            {"name": "whisper fine-tuned bc",
             "path": "inferences/grid_bc_turbo_v4",
             "restricted_vocab": False,},
        ]
        e = Evaluator(runs, save_at="wip")
        e.calculate_stoi_corr()

    runs = [
        {"name": "whisper fine-tuned bc",
         "path": "inferences/grid_bc_turbo_v4",
         "restricted_vocab": False, },
    ]
    e = Evaluator(runs, save_at="wip")
    e.foo(metric_column="entropy_kw_est")

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
        #grid
        "inferences/grid_bc_turbo_v4",
        "inferences/grid_bc_temp0.25_turbo_v4",
        #avsec
        "inferences/avsec_parakeet_defaut",
        #"inferences/avsec_turbo_default",
    ]
    configs = [load_config(Path(p) / "config.yaml") for p in paths]
    CorrelationManager.compare_correlations(configs)

if __name__ == '__main__':
    compare_graphs()
    #generate_correlation_summary()