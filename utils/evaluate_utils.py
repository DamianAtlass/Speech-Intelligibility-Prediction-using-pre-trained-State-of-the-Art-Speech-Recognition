import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
from pathlib import Path

import werpy
from matplotlib import pyplot as plt

def remove_nan(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    is_nan = torch.isnan(x) | torch.isnan(y)
    return x[~is_nan], y[~is_nan]


def calc_pearson_corr(x: torch.Tensor,
                      y: torch.Tensor,
                      output_path: Path | None,
                      name: str,
                      xlabel: str,
                      ylabel: str) -> None:

    x, y = remove_nan(x, y)
    # test for normality:
    n1 = stats.normaltest(x)
    n2 = stats.normaltest(y)

    regr = stats.linregress(x, y)
    print(regr)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "o", label="original data")
    plt.grid(True)
    title = f"Regression line and Pearson correlation coefficient of {name}"
    plt.suptitle(title)
    plt.title(f"Pearson's r: {regr.rvalue:.2f}, n ={len(x)}, p-value: {regr.pvalue:.2f}, Normality p-values: {n1.pvalue:.2f}, {n2.pvalue:.2f}")
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
    #plt.show()
    plt.close()

def calc_spearman_corr(x: torch.Tensor,
                      y: torch.Tensor,
                      output_path: Path | None,
                      name: str,
                       xlabel: str,
                       ylabel: str) -> None:
    x, y = remove_nan(x, y)

    x_ranked = stats.rankdata(x)
    y_ranked = stats.rankdata(y)

    # spearman corr == pearson corr of ranks
    regr = stats.linregress(x_ranked, y_ranked)

    plt.figure(figsize=(10, 5))
    plt.plot(x_ranked, y_ranked, "o", label="ranked data")
    plt.grid(True)
    title = f"Regression line and Spearman correlation coefficient of {name}"
    plt.suptitle(title)
    plt.title(f"Spearman's rho: {regr.rvalue:.2f}, n ={len(x)}, p-value: {regr.pvalue:.2f}")
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
    #plt.show()
    plt.close()


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
    #sort
    array = [a[~torch.isnan(a)] for a in array]
    if len(array)>1:
        order = torch.tensor([a.mean() for a in array])
        order = torch.argsort(order)

        array = [array[i] for i in order]
        x_label = [x_label[i] for i in order]


    fig, ax = plt.subplots(figsize=(4+len(array)*2,7))
    ax.grid()
    #tmp = ax.boxplot(array, showmeans=True)

    #plt.legend(loc='upper right')


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
               meanline=True,
             showmeans=True
             )
    plt.xticks(positions, x_label)


    ax.legend([tmp["means"][0]], ["Mean"], loc="upper right")
    if metric_name== "WER":
        plt.ylim(0,2)

    if output_path:
        plt.savefig(output_path/f'{title}.png')
    #plt.show()
    plt.close()

def plot_correlations(wers_human_kw, wers_machine_kw, avg_logprobs, config):
    # calc person correlation
    calc_pearson_corr(wers_human_kw,
                      wers_machine_kw,
                      config.output_path,
                      name=f"WER of human result and {config.model}({config.model_type})",
                      xlabel=f"WER of human results (only keywords)",
                      ylabel=f"WER ({config.model}, only keywords)")

    calc_pearson_corr(wers_human_kw,
                      avg_logprobs,
                      config.output_path,
                      name=f"the WER of human results and average log probability score of {config.model}({config.model_type})",
                      xlabel=f"WER of human results (only keywords)",
                      ylabel="average log probability score")

    # calc spearman correlation
    calc_spearman_corr(wers_human_kw,
                       wers_machine_kw,
                       config.output_path,
                       name=f"WER of human result and {config.model}({config.model_type})",
                       xlabel=f"WER of human results (only keywords)",
                       ylabel=f"WER ({config.model}, only keywords)")

    calc_spearman_corr(wers_human_kw,
                       avg_logprobs,
                       config.output_path,
                       name=f"the WER of human results and average log probability score of {config.model}({config.model_type})",
                       xlabel=f"WER of human results (only keywords)",
                       ylabel="average log probability score")