from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import stats as stats
from tqdm import tqdm

from utils.wer_needleman_wunsch import wer_needleman_wunsch

sorting_reverse = {
    "WER (machine)": True,
    "Logprob(per sequence)": False,
    "WER (machine, kw only)": True,
}

def plot_regr_line_for_pearson_corr(df: pd.DataFrame,
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


def plot_regr_line_for_spearman_corr(df: pd.DataFrame,
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

    plt.figure(figsize=(10, 7))
    plt.plot(x_ranked, y_ranked, "o", label="ranked data")
    plt.grid(True)
    title = f"Regression line and Spearman correlation coefficient of \n {name}"
    plt.suptitle(title)
    plt.title(f"\nSpearman's rho: {regr.rvalue:.2f}, n ={len(x_ranked)}, p-value: {regr.pvalue}, stderr: {regr.stderr:.4f}")
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
        order = torch.argsort(order, descending=sorting_reverse[metric_name])
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
    if "WER" in metric_name:
        max_y = None
        title += f", y-axis-limit={max_y}"
        plt.ylim(0, max_y)

    if len(array) > 1:
        title += ", sorted by means"
    plt.title(title)
    plt.ylabel(metric_name)
    ax.grid()
    plt.xticks(positions, x_label)
    plt.tight_layout()
    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")


    if output_path:
        plt.savefig(output_path/f'{figure_title}.png')
    #plt.show()
    plt.close()


def plot_wer_to_snr(df: pd.DataFrame,
                    shifting_attribute: str = "model_type",
                    shifting_attribute_label = None,
                    output_path: Path = None, ):

    list_shifting_attribute: list = list(df[shifting_attribute].unique())
    one_run_attribute: str = list_shifting_attribute[0]

    df_human_data = df[df[shifting_attribute]==one_run_attribute][["wers_human_kw", "snr"]]
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
    human_values = torch.tensor(human_values)*100

    machine_values: list = []
    for attr in list_shifting_attribute:
        df_attr = df[df[shifting_attribute] == attr]
        df_attr = df_attr.groupby(["snr"]).agg(avr_wer_machine=("wers_machine_kw", "mean")).reindex(np.sort(df["snr"].unique()))
        machine_values.append(df_attr["avr_wer_machine"].values * 100)


    positions = range(len(x_labels))
    plt.figure(figsize=[10, 5])
    plt.plot(positions, human_values, marker="o", label="human")
    for mv,l in zip(machine_values, list_shifting_attribute):
        plt.plot(positions, mv, marker="x", label=l)

    figure_title = f"WER of transcriptions from human data and {shifting_attribute_label or shifting_attribute}"
    plt.suptitle(figure_title)
    #plt.title(f"n={len(df)}") #falty
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel("Average WER in %")
    plt.grid()
    plt.legend()
    plt.ylim(0,100)
    if output_path:
        plt.savefig(output_path/f'{figure_title}.png')
    #plt.show()
    plt.close()


def plot_needleman_wunsch_wer_to_snr(df: pd.DataFrame,
                    shifting_attribute: str = "model_type",
                    shifting_attribute_label = None,
                    output_path: Path = None, ):

    list_shifting_attribute: list = list(df[shifting_attribute].unique())
    one_run_attribute: str = list_shifting_attribute[0]

    df_human_data = df[df[shifting_attribute]==one_run_attribute]

    human_values = []
    for snr in np.sort(df_human_data["snr"].unique()):
        df_snr = df_human_data[df_human_data["snr"]==snr]
        wer_snr = wer_needleman_wunsch(reference=df_snr["references_kw"].values, transcript=df_snr["human_transcripts_kw"].values)
        human_values.append(wer_snr)
    human_values = torch.Tensor(human_values) * 100
    del df_human_data
    x_labels = np.sort(df["snr"].unique())

    machine_values: list = []
    for attr in tqdm(list_shifting_attribute):
        df_attr = df[df[shifting_attribute] == attr]
        mv_temp = []
        for snr in np.sort(df_attr["snr"].unique()):
            df_snr = df_attr[df_attr["snr"] == snr]
            wer_snr = wer_needleman_wunsch(reference=df_snr["references_kw"].values,
                                           transcript=df_snr["machine_transcripts_kw"].values)
            mv_temp.append(wer_snr)
        machine_values.append(torch.tensor(mv_temp) *100)


    positions = range(len(x_labels))
    plt.figure(figsize=[10, 5])

    plt.plot(positions, human_values, marker="o", label="human")

    for mv,l in zip(machine_values, list_shifting_attribute):
        plt.plot(positions, mv, marker="x", label=l)

    figure_title = f"WER (Needleman-Wunsch) of transcriptions from human data and {shifting_attribute_label or shifting_attribute}"
    plt.suptitle(figure_title)
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel("Average WER (Needleman-Wunsch) in %")
    plt.ylim(0, 100)
    plt.grid()
    plt.legend()
    if output_path:
        plt.savefig(output_path/f'{figure_title}.png')
    #plt.show()
    plt.close()


def boxplot_corr_per_listener(df: pd.DataFrame,
                              correlate_to: str,
                              model: str,
                              model_type: str | list[str],
                              output_path: Path = None,
                              shifting_attribute = "model_type",
                              needlemanwunsch = False):
    wer_human_column = "wers_needlewunsch_human_kw" if needlemanwunsch else "wers_human_kw"

    list_shifting_attribute: list = list(df[shifting_attribute].unique())
    corr_arr = []
    p_val_arr = []
    for attr in list_shifting_attribute:
        df_model_type = df[df[shifting_attribute]==attr]
        df_model_type = df_model_type.dropna()

        corr_arr_tmp = []
        p_val_arr_tmp = []

        listeners = df_model_type["listener"].unique()
        for l in listeners:
            df_listener = df_model_type[df_model_type["listener"]==l]
            x = df_listener[wer_human_column]
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


    fig, ax = plt.subplots(figsize=(8 + len(list_shifting_attribute) * 0.7, 7))

    positions = range(1, len(list_shifting_attribute) + 1)

    tmp = ax.boxplot(corr_arr,
                     # notch=False,
                     positions=positions,
                     # meanline=True,
                     showmeans=True,
                     )
    d = {
        "wers_machine_kw": "WER for keywords",
        "wers_needlewunsch_machine_kw": "Needleman-Wunsch-WER for keywords",
        "avg_logprobs": "average log probability score (per sequence)",
        "avg_entropy": "mean entropy of all keywords in a sentence"
    }

    title = f"Spearman Correlation Coefficient of human {"Needleman-Wunsch-WER" if needlemanwunsch else "WER"} and \n {model}'s {d[correlate_to]} for each listener"
    plt.title(title +"\nand maximum p-value to the rounded 4th digit")
    plt.ylabel("Spearman Correlation Coefficient")
    ax.grid()
    x_label = [f"{t}\nmean={c.mean():.4f}\nmax(pvalue)={p.max():.4f}" for t,p, c in zip(list_shifting_attribute, p_val_arr, corr_arr)]
    plt.xticks(positions, x_label)
    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")
    plt.ylim(-1,1)


    if output_path:
        plt.savefig(output_path/f'{title}.png')
    #plt.show()
    plt.close()


