import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from pathlib import Path
import werpy
from matplotlib import pyplot as plt
import logging

from utils.cuda_utils import select_device
from utils.werpy_utils import additional_normalization, calculate_wers_with_norm

logger = logging.getLogger(__name__)
from utils.config_dataclasses import InferenceConfig


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


def calc_pearson_corr(df: pd.DataFrame,
                      name: str,
                      xlabel: str,
                      ylabel: str,
                      output_path: Path | None = None,
                      ) -> tuple:
    df = df.dropna()
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    # test for normality:
    normality_x = stats.normaltest(x)
    normality_y = stats.normaltest(y)

    regr = stats.linregress(x, y)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "o", label="original data")
    plt.plot(x,
             regr.intercept + regr.slope * x,
             "r",
             label=f"Regression line: y={regr.intercept:.2f}+{regr.slope:.2f}x")

    if ("WER" in xlabel) and ("WER" in ylabel):
        plt.ylim(0)
        plt.ylim(0)
    title = f"Regression line and Pearson correlation coefficient of {name}"
    plt.suptitle(title)
    plt.title(f"Pearson's r: {regr.rvalue:.2f}, n ={len(x)}, p-value: {regr.pvalue}, Normality p-values: {normality_x.pvalue:.2f}, {normality_y.pvalue:.2f}, stderr: {regr.stderr:.4f},")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path/f'{title}.png')
    #plt.show()
    plt.close()

    return regr.rvalue, regr.pvalue, normality_x.pvalue, normality_y.pvalue

def calc_spearman_corr(df: pd.DataFrame,
                      name: str,
                       xlabel: str,
                       ylabel: str,
                       output_path: Path | None = None) -> tuple:
    df = df.dropna()
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    x_ranked = stats.rankdata(x)
    y_ranked = stats.rankdata(y)
    del y,x
    # spearman corr == pearson corr of ranks
    regr = stats.linregress(x_ranked, y_ranked)

    plt.figure(figsize=(10, 5))
    plt.plot(x_ranked, y_ranked, "o", label="ranked data")
    plt.grid(True)
    title = f"Regression line and Spearman correlation coefficient of {name}"
    plt.suptitle(title)
    plt.title(f"Spearman's rho: {regr.rvalue:.2f}, n ={len(x_ranked)}, p-value: {regr.pvalue}, stderr: {regr.stderr:.4f}")
    plt.xlabel("ranked " + xlabel)
    plt.ylabel("ranked " + ylabel)
    if ("WER" in xlabel) and ("WER" in ylabel):
        plt.ylim(0)
        plt.ylim(0)
    plt.plot(x_ranked,
             regr.intercept + regr.slope * x_ranked,
             "r",
             label=f"Regression line: y={regr.intercept:.2f}+{regr.slope:.2f}x")
    plt.legend()

    if output_path:
        plt.savefig(output_path/f'{title}.png')
    #plt.show()
    plt.close()
    return regr.rvalue, regr.pvalue


def plot_metrics(array: list[pd.DataFrame],
                 title: str,
                 metric_name: str,
                 x_label: list[str],
                 output_path: Path | None
                 ) -> None:
    figure_title = title
    array = [a.dropna() for a in array]

    if len(array)>1:
        # sort
        order = torch.tensor([a.mean() for a in array])
        order = torch.argsort(order, descending=True)
        array = [array[i] for i in order]
        x_label = [x_label[i] for i in order]

    x_label = [x+f"\n mean = {a.mean():.2f}\nmedian = {a.median():.2f}\nn = {len(a)}" for (x,a) in zip(x_label, array)]

    fig, ax = plt.subplots(figsize=(6+len(array)*0.7,7))


    positions = range(1, len(x_label)+1)

    tmp = ax.boxplot(array,
                     #notch=False,
                     positions=positions,
                     #meanline=True,
                     showmeans=True,
             )
    if len(array) > 1:
        title += " sorted by means"
    plt.title(title)
    plt.ylabel(metric_name)
    ax.grid()
    plt.xticks(positions, x_label)
    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")

    if metric_name== "WER":
        plt.ylim(0,2)

    if output_path:
        plt.savefig(output_path/f'{figure_title}.png')
    #plt.show()
    plt.close()

def plot_correlations(df: pd.DataFrame, config):
    summary = []
    #wers_human_kw, wers_machine_kw, avg_logprobs
    # calc person correlation
    name = f"WER of human result and {config.model}({config.model_type})"
    r_val, p_val, x_normality_p, y_normality_p = calc_pearson_corr(df[["wers_human_kw", "wers_machine_kw"]],
                                                                      output_path=config.output_path,
                                                                      name=name,
                                                                      xlabel=f"WER of human results (only keywords)",
                                                                      ylabel=f"WER ({config.model}, only keywords)")
    summary.append({
        "metric": "person correlation",
        "of": name,
        "correlation_coefficient": f"{r_val:.10f}",
        "p_value": f"{p_val:.10f}",
        "x_normality_p_value": f"{x_normality_p:.10f}",
        "y_normality_p_value": f"{y_normality_p:.10f}",
    })

    name = f"the WER of human results and average log probability score of {config.model}({config.model_type})"
    r_val, p_val, x_normality_p, y_normality_p = calc_pearson_corr(df[["wers_human_kw", "avg_logprobs"]],
                                                                      output_path=config.output_path,
                                                                      name=name,
                                                                      xlabel=f"WER of human results (only keywords)",
                                                                      ylabel=f"average logprob ({config.model})")
    summary.append({
        "metric": "person correlation",
        "of": name,
        "correlation_coefficient": f"{r_val:.10f}",
        "p_value": f"{p_val:.10f}",
        "x_normality_p_value": f"{x_normality_p:.10f}",
        "y_normality_p_value": f"{y_normality_p:.10f}",
    })

    # calc spearman correlation
    name = f"WER of human result and {config.model}({config.model_type})"
    r_val, p_val = calc_spearman_corr(df[["wers_human_kw", "wers_machine_kw"]],
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

    name = f"the WER of human results and average log probability score of {config.model}({config.model_type})"
    r_val, p_val = calc_spearman_corr(df[["wers_human_kw", "avg_logprobs"]],
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

def plot_srt(df: pd.DataFrame, config):
    list_model_types: list = ([config.model_type] if isinstance(config.model_type, str) else config.model_type)
    one_model_type: str = list_model_types[0]

    df_human_data = df[df["model_type"]==one_model_type][["wers_human_kw", "snr"]]
    df_human_data_grouped = (
        df_human_data.groupby(["snr"])
        .agg(
            avr_wer_human=("wers_human_kw", "mean"),
        )
        .reindex(np.sort(df["snr"].unique()))
    )
    df = df.drop('wers_human_kw', axis=1)

    x_labels = np.sort(df["snr"].unique())
    human_values = df_human_data_grouped["avr_wer_human"].values

    machine_plots: list = []
    for t in list_model_types:
        df_tmp = df[df["model_type"] == t]
        df_tmp = df_tmp.groupby(["snr"]).agg(avr_wer_machine=("wers_machine_kw", "mean"), ).reindex(np.sort(df["snr"].unique()))
        machine_plots.append(df_tmp["avr_wer_machine"].values)


    positions = range(len(x_labels))
    plt.figure(figsize=[10, 5])
    plt.plot(positions, human_values, marker="o", label="human")
    for mp,l in zip(machine_plots, list_model_types):
        plt.plot(positions, mp, marker="x", label=l)

    figure_title = f"WER of transcription from human data and {config.model}({config.model_type})"
    plt.suptitle(figure_title)
    #plt.title(f"n={len(df)}") #falty
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel("Average WER")
    plt.grid()
    plt.legend()
    plt.savefig(config.output_path/f'{figure_title}.png')
    #plt.show()
    plt.close()

def calculate_corr_per_listener(df: pd.DataFrame,
                                config,
                                correlate_to: str):

    list_model_types: list = ([config.model_type] if isinstance(config.model_type, str) else config.model_type)
    corr_arr = []
    p_val_arr = []
    for t in list_model_types:
        df_model_type = df[df["model_type"]==t]
        df_model_type = df_model_type.dropna()

        corr_arr_tmp = []
        p_val_arr_tmp = []

        listeners = df_model_type["listener"].unique()
        for l in listeners:
            df_listener = df_model_type[df_model_type["listener"]==l]
            x = df_listener["wers_human_kw"]
            y = df_listener[correlate_to]
            x_ranked = stats.rankdata(x)
            y_ranked = stats.rankdata(y)
            del y, x

            # spearman corr == pearson corr of ranks
            regr = stats.pearsonr(x_ranked, y_ranked)
            corr_arr_tmp.append(regr.statistic)
            p_val_arr_tmp.append(regr.pvalue)

        corr_arr.append(torch.tensor(corr_arr_tmp))
        p_val_arr.append(torch.tensor(p_val_arr_tmp))


    fig, ax = plt.subplots(figsize=(6 + len(config.model_type) * 0.7, 7))

    positions = range(1, len(list_model_types) + 1)

    tmp = ax.boxplot(corr_arr,
                     # notch=False,
                     positions=positions,
                     # meanline=True,
                     showmeans=True,
                     )
    d = {
        "wers_machine_kw": "WER for keywords",
        "avg_logprobs": "average log probability score (per sequence)",
    }

    title = f"Spearman Correlation Coefficient between human WER and {config.model}'s {d[correlate_to]} for each listener"
    plt.title(title +"\nand maximum p-value to the rounded 4th digit")
    plt.ylabel("Pearson Correlation Coefficient")
    ax.grid()
    x_label = [f"{t}\nmean={c.mean():.4f}\nmax(pvalue)={p.max():.4f}" for t,p, c in zip(list_model_types, p_val_arr, corr_arr)]
    plt.xticks(positions, x_label)
    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")


    if config.output_path:
        plt.savefig(config.output_path/f'{title}.png')
    #plt.show()
    plt.close()

def evaluate_individual_run(config: InferenceConfig,
                            summary: list[dict],
                            df_single_run: pd.DataFrame,
                            device: torch.device):

    metrics = [df_single_run["avg_logprobs"], df_single_run["wers_machine"], df_single_run["wers_machine_kw"]]

    corr_summary = None

    if config.dataset_type != "grid":
        metrics.append(df_single_run["wers_human_kw"])
        corr_summary = plot_correlations(df_single_run, config)

        plot_srt(df_single_run[["wers_human_kw", "wers_machine_kw", "snr", "model_type"]], config)

        calculate_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine_kw", "model_type", "listener"]],
                                    config,
                                    correlate_to="wers_machine_kw")

        calculate_corr_per_listener(df_single_run[["wers_human_kw", "avg_logprobs", "model_type", "listener"]],
                                    config,
                                    correlate_to="avg_logprobs")

    for s_arr, metric in zip(summary, metrics):
        df_single_run["model_type"] = config.model_type
        plot_metrics([metric],
                     f"Average {s_arr["metric_name"]}s for {config.model}({config.model_type})",
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

    avg_logprobs = torch.Tensor(avg_logprobs).to(device)
    if dataset_type != "grid":
        wers_human_kw = calculate_wers_with_norm(reference=references_kw, hypothesis=human_transcripts_kw).to(device)

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

    }

    if dataset_type != "grid":
        data.update({
        "wers_human_kw": wers_human_kw.cpu(),
        "human_transcripts_kw": human_transcripts_kw,
        "listener": listener,
        "snr": snr,

        })

    df = pd.DataFrame(data)
    return summary, df
