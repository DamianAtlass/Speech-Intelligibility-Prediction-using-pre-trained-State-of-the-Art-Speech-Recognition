from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import stats as stats
from tqdm import tqdm
from typing import Literal, cast
from utils.wer_needleman_wunsch import wer_needleman_wunsch
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

grid_vocab = {
    "color": ['blue', 'green', 'red', 'white'], #4 items, index 1
    "letter": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z'], # 25 items, index 3
    "digit": ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero'] # 10 items, index 4
}

kw_labels = ["color", "letter", "digit"]
kw_colors = ["green", "blue", "red"]
kw_colors_short = ["g", "b", "r"]

sorting_reverse = {
    "wers_machine": True,
    "avg_logprobs": False,
    "wers_machine_kw": True,
}

labels_dict = {
    "average_macroscopic_entropy": "average (macroscopic) entropy",
    "wers_machine": "WER machine",
    "wers_machine_kw": "WER machine (keywords only)",
    'avg_logprobs': "Logprob (per sequence)",
    "machine_transcripts_len": "length of transcripts",
    "empty transcripts": "Amount of empty transcrips in %"
}

def wrap_text(text: str, max_chars: int = 75) -> str:
    """Insert line breaks so each line is at most max_chars characters."""
    words = text.split()
    lines = []
    current = ""

    for word in words:
        # Would adding this word exceed the limit?
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = word if not current else f"{current} {word}"

    if current:
        lines.append(current)

    return "\n".join(lines)

def join_kw_list(transcript_kw_or_similar: list[list[str]]) -> list[str]:
    """
    Args:
        transcript_kw_or_similar: list of keywords that should be joined to a list of string for calculation of the WER for example.
        Considers None values.

    Returns:
        list of joined strings
    """
    if isinstance(transcript_kw_or_similar[0], str):
        raise RuntimeError("Should not be a string!")
    return [" ".join(w for w in t if w is not None) for t in transcript_kw_or_similar]


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
    title = wrap_text(f"Regression line and Pearson correlation coefficient of {name}", 75)
    plt.suptitle(title)

    plt.title(f"Pearson's r: {regr.rvalue:.2f}, n ={len(x)}, p-value: {regr.pvalue}, Normality p-values: {normality_x.pvalue:.2f}, {normality_y.pvalue:.2f}, stderr: {regr.stderr:.4f}")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path/f'{title.replace("\n", "")}.png')
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
    title = f"Regression line and Spearman correlation coefficient of {name}"
    plt.suptitle(wrap_text(title))
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
        plt.savefig(output_path/f'{title.replace("\n", "")}.png')
    #plt.show()
    plt.close()
    return regr.rvalue, regr.pvalue


def plot_metrics(data: list[pd.Series],
                 x_label: list[str],
                 output_path: Path | None
                 ) -> None:
    column_name: str = data[0].name
    figure_title = f"Average {labels_dict[column_name]}"
    plot_title = figure_title
    data = [a.dropna() for a in data]

    if len(data)>1:
        # sort
        order = torch.tensor([a.mean() for a in data])
        order = torch.argsort(order, descending=sorting_reverse[column_name])
        data = [data[i] for i in order]
        x_label = [x_label[i] for i in order]

    x_label = [x +f"\n mean = {a.mean():.2f}\nmedian = {a.median():.2f}\nn = {len(a)}" for (x,a) in zip(x_label, data)]

    fig, ax = plt.subplots(figsize=(6 + len(data) * 0.7, 7))

    positions = range(1, len(x_label)+1)

    tmp = ax.boxplot(data,
                     #notch=False,
                     positions=positions,
                     #meanline=True,
                     showmeans=True,
                     )
    if "wer" in column_name:
        max_y = None
        figure_title += f", y-axis-limit={max_y}"
        plt.ylim(0, max_y)

    if len(data) > 1:
        figure_title += ", sorted by means"
    plt.title(wrap_text(figure_title))
    plt.ylabel(labels_dict[column_name])
    ax.grid()
    plt.xticks(positions, x_label)
    plt.tight_layout()
    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")


    if output_path:
        plt.savefig(output_path/f'{plot_title.replace("\n", "")}.png')
    #plt.show()
    plt.close()

def plot_wer_to_snr(
        df: pd.DataFrame,
        ref_col: Literal["references", "references_kw"],
        trans_col: Literal["machine_transcripts", "machine_transcripts_kw", "machine_trans_kw_from_time_align"],
        shifting_attribute: str = "model_type",
        shifting_attribute_label = None,
        output_path: Path = None, ):
    """
    Plot the WER for each SNR and add the human WER values for comparison.
    """

    if ("kw" in ref_col) != ("kw" in trans_col):
        raise RuntimeError("Sure those are the correct columns?")

    list_shifting_attribute: list = list(df[shifting_attribute].unique())
    one_run_attribute: str = list_shifting_attribute[0]

    #plot human values
    df_human_data = df[df[shifting_attribute]==one_run_attribute]

    human_values = []
    for snr in np.sort(df_human_data["snr"].unique()):
        df_snr = df_human_data[df_human_data["snr"]==snr]
        wer_snr = wer_needleman_wunsch(references=join_kw_list(df_snr["references_kw"].values),
                                       transcripts=join_kw_list(df_snr["human_transcripts_kw"].values))
        human_values.append(wer_snr)
    human_values = torch.Tensor(human_values) * 100
    del df_human_data
    x_labels = np.sort(df["snr"].unique())

    #plot machine values
    machine_values: list = []
    for attr in tqdm(list_shifting_attribute):
        df_attr = df[df[shifting_attribute] == attr]
        mv_temp = []
        for snr in np.sort(df_attr["snr"].unique()):
            df_snr = df_attr[df_attr["snr"] == snr]

            def get_values(col):
                values = df_snr[col].values
                return join_kw_list(values) if "kw" in col else values

            references = get_values(ref_col)
            transcripts = get_values(trans_col)

            wer_snr = wer_needleman_wunsch(references=references,
                                           transcripts=transcripts)
            mv_temp.append(wer_snr)
        machine_values.append(torch.tensor(mv_temp) * 100)


    positions = range(len(x_labels))
    plt.figure(figsize=[10, 5])

    plt.plot(positions, human_values, marker="o", label="human")

    for mv,l in zip(machine_values, list_shifting_attribute):
        plt.plot(positions, mv, marker="x", label=l)

    align_info_str = ", derived from time alignments" if "align" in trans_col else ""
    kw_info_str = f" (keyword only{align_info_str})" if "kw" in ref_col else ""
    figure_title = f"WER of human transcription vs machine transcripts{kw_info_str} by {shifting_attribute_label or shifting_attribute}"
    plt.suptitle(wrap_text(figure_title))
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel("Average WER in %")
    plt.ylim(0, max([100, max([max(l) for l in machine_values])]))
    plt.grid()
    plt.legend()
    if output_path:
        plt.savefig(output_path/f'{figure_title.replace("\n", "")}.png')
    #plt.show()
    plt.close()

def plot_x_to_snr(df: pd.DataFrame,
                    plotting_attribute: str,
                    shifting_attribute: str = "model_type",
                    shifting_attribute_label: str = None,
                    output_path: Path|None = None, ):

    """
    Plot a specific column to all SNRs, without human WER data for comparison.
    """
    list_shifting_attribute: list = list(df[shifting_attribute].unique())
    x_labels = np.sort(df["snr"].unique())

    values: list = []
    for attr in tqdm(list_shifting_attribute):
        df_attr = df[df[shifting_attribute] == attr]
        values_per_df = []
        for snr in np.sort(df_attr["snr"].unique()):
            df_snr = df_attr[df_attr["snr"] == snr]
            if plotting_attribute == "empty transcripts":
                value = sum(df_snr["machine_transcripts"] == "") / len(df_snr)
            else:
                v = torch.tensor(df_snr[plotting_attribute].values)
                value = torch.mean(v[~v.isnan()])
            values_per_df.append(value)
        values.append(torch.tensor(values_per_df))


    positions = range(len(x_labels))
    plt.figure(figsize=[10, 5])

    for mv,l in zip(values, list_shifting_attribute):
        plt.plot(positions, mv, marker="x", label=l)

    figure_title = f"{labels_dict[plotting_attribute]} for {shifting_attribute_label or shifting_attribute}"
    plt.suptitle(figure_title)
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel(labels_dict[plotting_attribute])
    plt.grid()
    plt.ylim(0)

    plt.legend()
    if output_path:
        plt.savefig(output_path/f'{figure_title.replace("\n", "")}.png')
    #plt.show()
    plt.close()


def plot_microscopic_x_to_snr(df: pd.DataFrame,
                              col_name: Literal["entropies_kw", "entropies_kw_from_time_align", "tad_kw"],
                              value_label: str,
                              y_axis_label: str = None,
                              shifting_attribute: str = "model_type",
                              shifting_attribute_label = None,
                              output_path: Path = None, ):
    """
    Regular plot for unspecific columns to (line)plot for each SNR.
    """
    if "kw" not in col_name:
        raise ValueError("This plot is for microscopic plotting (per kw)!")

    list_shifting_attribute: list = list(df[shifting_attribute].unique())

    x_labels = np.sort(df["snr"].unique())

    values_means: list = []
    for attr in tqdm(list_shifting_attribute):
        df_attr = df[df[shifting_attribute] == attr]
        values_per_snr = []
        for snr in np.sort(df_attr["snr"].unique()):

            df_snr = df_attr[df_attr["snr"] == snr]
            values_current_snr = []
            for kw in range(3):

                values_for_this_kw = df_snr[col_name].apply(lambda x: x[kw])
                values_for_this_kw = values_for_this_kw[values_for_this_kw.notnull()]
                tmp2 = np.mean(values_for_this_kw)
                assert tmp2 != np.nan
                values_current_snr.append(tmp2)
            values_per_snr.append(values_current_snr)
        values_means.append(values_per_snr)

    positions = range(len(x_labels))
    plt.figure(figsize=[10, 5])
    colors = ["b", "g", "c", "r", ]
    line_type = ['-', '--', ':', '-.']
    for mv,l,lt in zip(values_means, list_shifting_attribute, line_type):
        for kw, c in zip(range(3), kw_colors_short):
            plt.plot(positions, [o[kw] for o in mv], marker="x", color=c, ls=lt, label=f"{l} | {kw_labels[kw]}")


    figure_title = f"Average {value_label} of keywords {"(derived from time alignments)" if "from_time_align" in col_name else ""} for {shifting_attribute_label or shifting_attribute}"
    plt.suptitle(wrap_text(figure_title))
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel(f"microscopic {y_axis_label or value_label}")
    plt.ylim(0)
    plt.grid()
    plt.legend()
    if output_path:
        plt.savefig(output_path/f'{figure_title}.png')
    #plt.show()
    plt.close()

from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

def boxplot_microscopic_x_to_snr(
        df: pd.DataFrame,
        col_name: Literal["entropies_kw", "entropies_kw_from_time_align", "tad_kw"],
        value_label: str,
        special_metric: Literal["correlation"]|None = None,
        y_axis_label: str = None,
        output_path: Path = None):
    """
    Create 3 boxplots per SNR and group values by the keyword.
    CAN be used to plot correlation to human_transcripts_kw per keyword and SNR, but doesn't have to.
    """

    if not special_metric and "kw" not in col_name:
        raise ValueError("This plot is for microscopic plotting (per kw)!")

    x_labels = np.sort(df["snr"].unique())

    values_per_snr = []
    for snr in np.sort(df["snr"].unique()):

        df_snr = df[df["snr"] == snr]
        values_current_snr = []
        for i_kw in range(3):
            values_keyword = []
            keywords: list[str] = grid_vocab[kw_labels[i_kw]]
            for keyword in keywords:
                df_snr_keyword = df_snr[
                    df_snr["references_kw"].str[i_kw].eq(keyword)
                ]
                #calculate metric to plot
                if not special_metric:
                    values_for_this_keyword = df_snr_keyword[col_name].apply(lambda x: x[i_kw])
                    values_for_this_keyword = values_for_this_keyword[values_for_this_keyword.notnull()]

                    tmp2 = np.mean(values_for_this_keyword)
                    assert tmp2 != np.nan
                    values_keyword.append(tmp2)
                elif special_metric == "correlation":
                    x = df_snr_keyword["tad_kw"].apply(lambda x: x[i_kw])

                    y1 = df_snr_keyword[col_name].apply(lambda x: x[i_kw])
                    y2 = df_snr_keyword["human_transcripts_kw"].apply(lambda x: x.split()[i_kw])
                    y = (y1!=y2).astype(int) # wer with human results basically
                    x_ranked = stats.rankdata(x)
                    y_ranked = stats.rankdata(y)
                    del y, x

                    # spearman corr == pearson corr of ranks
                    regr = stats.pearsonr(x_ranked, y_ranked)
                    values_keyword.append(regr.statistic)
                    #regr.pvalue
                else:
                    raise NotImplementedError
            values_current_snr.append(values_keyword)
        values_per_snr.append(values_current_snr)

    space_between_plots = 2
    a = 3 + space_between_plots

    positions = torch.Tensor(range(len(x_labels)))*a+1
    plt.figure(figsize=[10, 5])

    #https://stackoverflow.com/questions/16592222/how-to-create-grouped-boxplots
    def setBoxColors(bp):
        for i in range(3):
            c = kw_colors[i]
            setp(bp['boxes'][i], color=c)
            setp(bp['caps'][i*2], color=c)
            setp(bp['caps'][i*2+1], color=c)
            setp(bp['whiskers'][i*2], color=c)
            setp(bp['whiskers'][i*2+1], color=c)
            setp(bp['fliers'][i], markeredgecolor=c)
            setp(bp['medians'][i], color=c)

    for i, snr_values in enumerate(values_per_snr):
        pos = list(range(i*a, (i*a)+a))[:-space_between_plots]
        bp = boxplot(snr_values, positions = pos, widths = 0.6)
        setBoxColors(bp)

    # create lines, use them for the legend and make them invisible afterwards
    hG, = plot([1, 1], 'g-')
    hB, = plot([1, 1], 'b-')
    hR, = plot([1, 1], 'r-')

    legend((hG, hB, hR), (kw_labels[0], kw_labels[1], kw_labels[2]))
    hB.set_visible(False)
    hB.set_visible(False)
    hG.set_visible(False)

    figure_title = f"Average {value_label} of keywords {"(derived from time alignments)" if "from_time_align" in col_name else ""}grouped by reference keywords"
    plt.suptitle(wrap_text(figure_title))
    plt.xticks(positions, x_labels)
    plt.xlabel("SNR")
    plt.ylabel(f"microscopic {y_axis_label or value_label}")
    plt.ylim(0)
    plt.grid()

    if output_path:
        plt.savefig(output_path/f'{figure_title}.png')
    plt.close()

def boxplot_corr_per_listener(df: pd.DataFrame,
                              correlate_to: str,
                              model: str,
                              model_type: str | list[str],
                              output_path: Path = None,
                              shifting_attribute = "model_type"):
    """
    Boxplots grouped by listeners. May need an update.
    """

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
            x = df_listener[correlate_to]
            y = df_listener["wers_human_kw"]
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
        "wers_machine": "WER for the whole sequence",
        "avg_logprobs": "average log probability score (per sequence)",
        "average_macroscopic_entropy": "average (macroscopic) entropy of all words in a sentence"
    }

    title = f"Spearman Correlation Coefficient of human WER and {model}'s {d[correlate_to]} for each listener"
    plot_title = title +" and maximum p-value to the rounded 4th digit"
    plt.title(wrap_text(plot_title, 75))

    plt.ylabel("Spearman Correlation Coefficient")
    ax.grid()
    x_label = [f"{t}\nmean={c.mean():.4f}\nmax(pvalue)={p.max():.4f}" for t,p, c in zip(list_shifting_attribute, p_val_arr, corr_arr)]
    plt.xticks(positions, x_label)
    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")
    plt.ylim(-1,1)

    if output_path:
        plt.savefig(output_path/f'{title.replace("\n", "")}.png')
    #plt.show()
    plt.close()

def boxplot_microscopic_special_metric_per_keyword(
        df: pd.DataFrame,
        col_name: Literal["entropies_kw", "entropies_kw_from_time_align", "tad_kw"],
        col_compare_against_ref_kw: Literal["estimated_transcript_kw", "machine_trans_kw_from_time_align", "human_transcripts_kw"],  #kw column, estimated_transcript_kw for calibration
        special_metric: Literal["spearman_correlation", "mutual_information"] = "spearman_correlation",
        col_title: Literal["entropy", "TAD"]|str = "entropy",
        output_path: Path|None = None):
    """
    3 boxplots for either spearman correlation or mutual information
    """

    tmp_labels_dict = {
        "human_transcripts_kw": "listener's",
        "estimated_transcript_kw": " machine's", # specifically for calibration
        "machine_trans_kw_from_time_align": "time-alignment-derived machine's, ",  # specifically for calibration
    }
    cali = " (calibration)" if "machine_trans_kw" in col_compare_against_ref_kw else ""
    metric_name = {
        "spearman_correlation": "Spearman Correlation",
        "mutual_information": "Mutual Information"
    }

    # the speech intelligibility is basically the wer between human transcripts and reference.
    # we want to measure a correlation between that, and the entropies. no need for  machine_trans_kw(_from_time_align) here!
    # To check the calibration, we measure correlation between the wer of the machine (ref_kw vs machine_trans_kw).

    value_array = []
    p_val_arr = []
    for kw_idx in tqdm(range(3)):
        value_array_per_kw = []
        p_val_arr_per_kw = []
        for kw in grid_vocab[kw_labels[kw_idx]]:
            df_keyword = df[
                df["references_kw"].str[kw_idx].eq(kw)
            ]

            ref_kw = df_keyword["references_kw"].map(lambda x: x[kw_idx])

            keywords = df_keyword[col_compare_against_ref_kw].map(lambda x: x[kw_idx])

            x = df_keyword[col_name].map(lambda x: x[kw_idx])
            y = (ref_kw!=keywords).astype(int)  # basically the WER

            filter = x.isna()
            x = torch.from_numpy(np.array(x.astype(float))[~filter])
            y = y[~filter]
            if special_metric == "spearman_correlation":

                x_ranked = stats.rankdata(x)
                y_ranked = stats.rankdata(y)
                del y, x

                # spearman corr == pearson corr of ranks
                regr = stats.pearsonr(x=x_ranked, y=y_ranked)
                value_array_per_kw.append(regr.statistic)
                p_val_arr_per_kw.append(regr.pvalue)

            elif special_metric == "mutual_information":
                mi = mutual_info_classif(X=torch.Tensor(x).reshape(-1, 1), y=y)
                value_array_per_kw.append(mi[0])
            else:
                raise NotImplementedError

        value_array.append(torch.tensor(value_array_per_kw))
        p_val_arr.append(torch.tensor(p_val_arr_per_kw))

    fig, ax = plt.subplots(figsize=(9, 7))

    positions = range(1, 3 + 1)


    tmp = ax.boxplot(value_array,
                     # notch=False,
                     positions=positions,
                     # meanline=True,
                     showmeans=True,
                     )

    title = f"{metric_name[special_metric]} between the {tmp_labels_dict[col_compare_against_ref_kw]} word-level WER and whisper's token-level {col_title} for each keyword{cali}"
    plot_title = title + (" and maximum p-value to the rounded 4th digit" if special_metric == "spearman_correlation" else "")
    plt.title(wrap_text(plot_title))
    plt.ylabel("Spearman Correlation Coefficient" if special_metric == "spearman_correlation" else "Mutual Information")
    ax.grid()
    if special_metric == "spearman_correlation":
        x_label = [f"{t}\nmean={c.mean():.4f}\nmax(pvalue)={p.max():.4f}" for t, p, c in
                   zip(kw_labels, p_val_arr, value_array)]
    else:
        x_label = [f"{t}\nmean={c.mean():.4f}\n" for t, c in
                   zip(kw_labels, value_array)]

    plt.xticks(positions, x_label)
    ax.legend([tmp["means"][0], tmp["medians"][0]], ["Means", "Medians"], loc="upper right")
    plt.ylim(-1, 1)


    if output_path:
        plt.savefig(output_path/f'{title.replace("\n", "")}.png')
    #plt.show()
    plt.close()


