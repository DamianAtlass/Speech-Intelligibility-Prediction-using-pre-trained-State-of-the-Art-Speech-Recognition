import numpy as np
import matplotlib.pyplot as plt
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


def calc_pearson_corr(x: torch.Tensor,
                      y: torch.Tensor,
                      name: str,
                      xlabel: str,
                      ylabel: str,
                      output_path: Path | None = None,
                      ) -> tuple:
    x, y = x.cpu(), y.cpu()
    x, y = remove_nan(x, y)
    # test for normality:
    normality_x = stats.normaltest(x)
    normality_y = stats.normaltest(y)

    regr = stats.linregress(x, y)
    logger.info(regr)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "o", label="original data")
    plt.grid(True)
    title = f"Regression line and Pearson correlation coefficient of {name}"
    plt.suptitle(title)
    plt.title(f"Pearson's r: {regr.rvalue:.2f}, n ={len(x)}, p-value: {regr.pvalue}, Normality p-values: {normality_x.pvalue:.2f}, {normality_y.pvalue:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ("WER" in xlabel) and ("WER" in ylabel):
        plt.ylim(0)
        plt.ylim(0)
    plt.plot(x,
             regr.intercept + regr.slope * x,
             "r",
             label=f"Regression line: y={regr.intercept:.2f}+{regr.slope:.2f}x")
    plt.legend()

    if output_path:
        plt.savefig(output_path/f'{title}.png')
    plt.show()
    plt.close()

    return regr.rvalue, regr.pvalue, normality_x.pvalue, normality_y.pvalue

def calc_spearman_corr(x: torch.Tensor,
                      y: torch.Tensor,
                      name: str,
                       xlabel: str,
                       ylabel: str,
                       output_path: Path | None = None) -> tuple:
    x, y = x.cpu(), y.cpu()
    x, y = remove_nan(x, y)

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
    plt.title(f"Spearman's rho: {regr.rvalue:.2f}, n ={len(x_ranked)}, p-value: {regr.pvalue}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
    plt.show()
    plt.close()
    return regr.rvalue, regr.pvalue

def get_only_keywords(string) -> str:
    """
    Return only the words at the keyword indices
    """
    string = werpy.normalize(string)
    keywords_index = [1, 3, 4]
    string = string.split(" ")
    string = [s for i,s in enumerate(string) if i in keywords_index]
    return " ".join(string)


def plot_metrics(array: list[torch.Tensor],
                 title: str,
                 metric_name: str,
                 x_label: list[str],
                 output_path: Path | None
                 ) -> None:
    figure_title = title
    #sort
    array = [a[~torch.isnan(a)] for a in array]
    if len(array)>1:
        order = torch.tensor([a.mean() for a in array])
        order = torch.argsort(order, descending=True)

        array = [array[i] for i in order]
        x_label = [x_label[i] for i in order]

    x_label = [x+f"\n mean = {a.mean():.2f}\nmedian = {a.median():.2f}\nn = {len(a)}" for (x,a) in zip(x_label, array)]

    fig, ax = plt.subplots(figsize=(4+len(array)*2,7))
    ax.grid()
    #tmp = ax.boxplot(array, showmeans=True)

    #plt.legend(loc='upper right')

    if len(array) > 1:
        title += " sorted by means"
    plt.title(title)
    #plt.xlabel(x_label)
    plt.ylabel(metric_name)

    positions = range(1, len(x_label)+1)
    #print(list(positions))
    #print(len(array))
    array = [a.cpu() for a in array]
    tmp = ax.boxplot(array,
                     #notch=False,
                     positions=positions,
                     #meanline=True,
                     showmeans=True,

             )
    plt.xticks(positions, x_label)


    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")

    if metric_name== "WER":
        plt.ylim(0,2)

    if output_path:
        plt.savefig(output_path/f'{figure_title}.png')
    plt.show()
    plt.close()

def plot_correlations(wers_human_kw, wers_machine_kw, avg_logprobs, config):
    summary = []
    # calc person correlation
    name = f"WER of human result and {config.model}({config.model_type})"
    r_val, p_val, x_normality_p, y_normality_p = calc_pearson_corr(wers_human_kw,
                      wers_machine_kw,
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

    # calc spearman correlation
    name = f"WER of human result and {config.model}({config.model_type})"
    r_val, p_val = calc_spearman_corr(wers_human_kw,
                       wers_machine_kw,
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
    r_val, p_val = calc_spearman_corr(wers_human_kw,
                       avg_logprobs,
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