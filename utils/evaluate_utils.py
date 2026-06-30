import json
from typing import Tuple, List

import pandas as pd
import torch
from pathlib import Path
import werpy
import logging

from utils.plotting_utils import plot_regr_line_for_spearman_corr, plot_metrics, plot_wer_to_snr, \
    plot_needleman_wunsch_wer_to_snr, boxplot_corr_per_listener
from utils.werpy_utils import additional_normalization, calculate_wers_with_norm
from utils.wer_needleman_wunsch import wer_needleman_wunsch

logger = logging.getLogger(__name__)
from utils.config_dataclasses import InferenceConfig

sorting_reverse = {
    "WER (machine)": True,
    "Logprob(per sequence)": False,
    "WER (machine, kw only)": True,
}

def remove_nan(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    is_nan = torch.isnan(x) | torch.isnan(y)
    return x[~is_nan], y[~is_nan]

def get_only_keywords(string) -> str:
    """
    Return only the words at the keyword indices
    """
    string = werpy.normalize(string)
    keywords_index = [1, 3, 4]
    string = string.split(" ")
    string = [s for i,s in enumerate(string) if i in keywords_index]
    return " ".join(string)


def plot_regr_lines(df: pd.DataFrame, config):
    summary = []
    #wers_human_kw, wers_machine_kw, avg_logprobs
    # calc person correlation

    # name = f"WER of human result and {config.model}({config.model_type})"
    # r_val, p_val, x_normality_p, y_normality_p = calc_pearson_corr(df[["wers_human_kw", "wers_machine_kw"]],
    #                                                                   output_path=config.output_path,
    #                                                                   name=name,
    #                                                                   xlabel=f"WER of human results (only keywords)",
    #                                                                   ylabel=f"WER ({config.model}, only keywords)")
    # summary.append({
    #     "metric": "person correlation",
    #     "of": name,
    #     "correlation_coefficient": f"{r_val:.10f}",
    #     "p_value": f"{p_val:.10f}",
    #     "x_normality_p_value": f"{x_normality_p:.10f}",
    #     "y_normality_p_value": f"{y_normality_p:.10f}",
    # })

    # name = f"the WER of human results and average log probability score of {config.model}({config.model_type})"
    # r_val, p_val, x_normality_p, y_normality_p = calc_pearson_corr(df[["wers_human_kw", "avg_logprobs"]],
    #                                                                   output_path=config.output_path,
    #                                                                   name=name,
    #                                                                   xlabel=f"WER of human results (only keywords)",
    #                                                                   ylabel=f"average logprob ({config.model})")
    # summary.append({
    #     "metric": "person correlation",
    #     "of": name,
    #     "correlation_coefficient": f"{r_val:.10f}",
    #     "p_value": f"{p_val:.10f}",
    #     "x_normality_p_value": f"{x_normality_p:.10f}",
    #     "y_normality_p_value": f"{y_normality_p:.10f}",
    # })

    # calc spearman correlation
    name = f"the WER of human result and {config.model}({config.model_type})"
    r_val, p_val = plot_regr_line_for_spearman_corr(df[["wers_human_kw", "wers_machine_kw"]],
                                                    output_path=config.output_path,
                                                    name=name,
                                                    xlabel=f"WER of human results (only keywords)",
                                                    ylabel=f"WER ({config.model}, only keywords)")
    summary.append({
        "metric": "spearman correlation",
        "of": name,
        "correlation_coefficient": f"{r_val:.10f}",
        "p_value": f"{p_val:.10f}",
    })

    name = f"the Needleman-Wunsch-WER of human results and {config.model}({config.model_type})"
    r_val, p_val = plot_regr_line_for_spearman_corr(df[["wers_needlewunsch_human_kw", "wers_needlewunsch_machine_kw"]],
                                                    output_path=config.output_path,
                                                    name=name,
                                                    xlabel=f"Needleman-Wunsch-WER of human results (only keywords)",
                                                    ylabel=f"Needleman-Wunsch-WER ({config.model}, only keywords)")

    summary.append({
        "metric": "spearman correlation",
        "of": name,
        "correlation_coefficient": f"{r_val:.10f}",
        "p_value": f"{p_val:.10f}",
    })

    name = f"the WER of human results and average log probability score of {config.model}({config.model_type})"
    r_val, p_val = plot_regr_line_for_spearman_corr(df[["wers_human_kw", "avg_logprobs"]],
                                                    output_path=config.output_path,
                                                    name=name,
                                                    xlabel=f"WER of human results (only keywords)",
                                                    ylabel="average log probability score")
    summary.append({
        "metric": "spearman correlation",
        "of": name,
        "correlation_coefficient": f"{r_val:.10f}",
        "p_value": f"{p_val:.10f}",
    })

    return summary


def evaluate_individual_run(config: InferenceConfig,
                            summary: list[dict],
                            df_single_run: pd.DataFrame,
                            device: torch.device):

    metrics = [df_single_run["avg_logprobs"], df_single_run["wers_machine"], df_single_run["wers_machine_kw"]]

    corr_summary = None

    if config.dataset_type != "grid":
        metrics.append(df_single_run["wers_human_kw"])
        corr_summary = plot_regr_lines(df_single_run, config)

        plot_wer_to_snr(df=df_single_run[["wers_human_kw", "wers_machine_kw", "snr", "model_type"]],
                        shifting_attribute="model_type",
                        output_path=config.output_path)

        plot_needleman_wunsch_wer_to_snr(df=df_single_run[["human_transcripts_kw", "machine_transcripts_kw", "snr", "references_kw", "model_type"]],
                                         shifting_attribute="model_type",
                                         output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine_kw", "model_type", "listener"]],
                                  correlate_to="wers_machine_kw",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_needlewunsch_human_kw", "wers_needlewunsch_machine_kw", "model_type", "listener"]],
                                  correlate_to="wers_needlewunsch_machine_kw",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path,
                                  needlemanwunsch=True)

        boxplot_corr_per_listener(df_single_run[["wers_needlewunsch_human_kw", "avg_logprobs", "model_type", "listener"]],
                                  correlate_to="avg_logprobs",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path,
                                  needlemanwunsch=True)

    for s_arr, metric in zip(summary, metrics):
        df_single_run["model_type"] = config.model_type
        plot_metrics([metric],
                     f"Average {s_arr["metric_name"]} for {config.model}({config.model_type})",
                     s_arr["metric_name"],
                     [config.model_type],
                     config.output_path)

    with open(config.output_path / "summary.json", 'w') as f:
        json.dump({"summary:": summary, "correlation:": corr_summary if corr_summary else None}, f, indent=4)


def get_data(output_path: Path,
             dataset_type: str,
             device: torch.device) -> Tuple[List[dict], pd.DataFrame]:

    data_path = output_path / "data"

    avg_logprobs = []
    references = []
    machine_transcripts = []
    human_transcripts_kw = []
    snr = []
    listener = []
    audio_paths = []

    for file in data_path.iterdir():
        with open(file) as f:
            json_file = json.load(f)

            if json_file["prediction_result"]["text"] == "" : #nothing recognized!
                avg_logprobs.append(torch.nan)
                machine_transcripts.append("")
            else:
                avg_logprobs.append(json_file["prediction_result"]["segments"][0]["avg_logprob"])
                machine_transcripts.append(json_file["prediction_result"]["text"])

            references.append(json_file["sentence"])
            audio_paths.append(json_file["audio_path"])
            if dataset_type != "grid":
                human_transcripts_kw.append(json_file["human_recognized_words"])
                snr.append(int(json_file["snr_db"]))
                listener.append(json_file["listener"])


    machine_transcripts = additional_normalization(machine_transcripts)
    machine_transcripts_kw = [get_only_keywords(o) for o in machine_transcripts]

    human_transcripts_kw = additional_normalization(human_transcripts_kw)

    references_kw = [get_only_keywords(o) for o in references]

    wers_machine = calculate_wers_with_norm(reference=references, hypothesis=machine_transcripts).to(device)
    wers_machine_kw = calculate_wers_with_norm(reference=references_kw, hypothesis=machine_transcripts_kw).to(device)

    wers_needlewunsch_machine_kw = [wer_needleman_wunsch([r], [t]) for r,t in zip(references_kw, machine_transcripts_kw)]

    avg_logprobs = torch.Tensor(avg_logprobs).to(device)
    if dataset_type != "grid":
        wers_human_kw = calculate_wers_with_norm(reference=references_kw, hypothesis=human_transcripts_kw).to(device)
        wers_needlewunsch_human_kw = [wer_needleman_wunsch([r], [t]) for r,t in zip(references_kw, human_transcripts_kw)]

    summary = []
    metric_names = ["Logprob(per sequence)", "WER (machine)", "WER (machine, kw only)"]
    metrics = [avg_logprobs, wers_machine, wers_machine_kw]
    if dataset_type != "grid":
        metric_names.append("WER (human study, kw only)")
        metrics.append(wers_human_kw)

    for n, m in zip(metric_names, metrics):
        summary.append({
            "metric_name": n,
            "mean": m.mean().item(),
            "median": m.median().item(),
            "std": m.std().item(),
            "n": m.shape[0]
        })

    data = {
        "avg_logprobs": avg_logprobs.cpu(),
        "references": references,
        "references_kw": references_kw,
        "wers_machine": wers_machine.cpu(),
        "wers_machine_kw": wers_machine_kw.cpu(),
        "machine_transcripts": machine_transcripts,
        "machine_transcripts_kw": machine_transcripts_kw,
        "audio_paths": audio_paths,
        "wers_needlewunsch_machine_kw": wers_needlewunsch_machine_kw,

    }

    if dataset_type != "grid":
        data.update({
        "wers_human_kw": wers_human_kw.cpu(),
        "human_transcripts_kw": human_transcripts_kw,
        "listener": listener,
        "snr": snr,
        "wers_needlewunsch_human_kw": wers_needlewunsch_human_kw,

        })

    df = pd.DataFrame(data)
    return summary, df
