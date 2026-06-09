import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import torch
from pathlib import Path

import werpy
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)

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
    df_grouped = (
        df.groupby("snr")
          .agg(
            avr_wer_human=("wers_human_kw", "mean"),
            avr_wer_machine=("wers_machine_kw", "mean"),
        )
          .reindex(np.sort(df["snr"].unique()))
    )

    x_labels = df_grouped.index.tolist()
    y1_values = df_grouped["avr_wer_human"].values
    y2_values = df_grouped["avr_wer_machine"].values

    positions = range(len(x_labels))

    plt.plot(positions, y1_values, marker="o", label="human")
    plt.plot(positions, y2_values, marker="x", label="machine")

    figure_title = f"WER of transcription from human data and {config.model}({config.model_type})"
    plt.suptitle(figure_title)
    plt.title(f"n={len(df)}")
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel("Average WER")
    plt.grid()
    plt.legend()
    plt.savefig(config.output_path/f'{figure_title}.png')
    #plt.show()
    plt.close()
