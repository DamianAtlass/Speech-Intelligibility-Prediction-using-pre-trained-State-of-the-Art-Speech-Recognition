import json
from typing import Tuple, List

import pandas as pd
import torch
from pathlib import Path
import werpy
import logging
from scipy.stats import entropy
from whisper.tokenizer import get_tokenizer
import numpy as np
import string

from utils.plotting_utils import plot_regr_line_for_spearman_corr, plot_metrics, \
    plot_needleman_wunsch_wer_to_snr, boxplot_corr_per_listener, plot_entropy
from utils.werpy_utils import normalize
from utils.wer_needleman_wunsch import wer_needleman_wunsch, wer_needleman_wunsch_per_sample, _needlemann_wunsch

logger = logging.getLogger(__name__)
from utils.config_dataclasses import InferenceConfig


def remove_nan(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    is_nan = torch.isnan(x) | torch.isnan(y)
    return x[~is_nan], y[~is_nan]

def find_ordered_indices(transcript: list, keywords_to_find: list) -> list[int]:
    indices = []
    j = 0  # pointer for keywords

    assert [transcript.index(s) for s in keywords_to_find if s is not None] or True # no exception being thrown is assert enough

    for i, val in enumerate(transcript):
        if j < len(keywords_to_find):
            if keywords_to_find[j] is None:
                indices.append(None)
                j+=1
                continue

            if val == keywords_to_find[j]:
                indices.append(i)
                j += 1
                if j == len(keywords_to_find):
                    break
    rest = len(keywords_to_find) - len(indices)
    indices.extend([None for _ in range(rest)])
    assert len(indices) == len(keywords_to_find)
    return indices



    return indices if j == len(keywords_to_find) else None

def get_only_keywords(string) -> str:
    """
    Return only the words at the keyword indices. Only use when you expect the correct length of 6 words.
    """
    string = string.split()
    if len(string) != 6:
        raise ValueError(f"Expected 6 words, got {len(string)}")

    keywords_index = [1, 3, 4]

    string = [s for i,s in enumerate(string) if i in keywords_index]
    return " ".join(string)

def get_only_keywords_using_alignments(reference: list[str], transcript: list[str], return_idx=False) -> list[str] | list[int]:
    """
    Use alignment to return only the words at the keywords of string at the keyword indices. Input should be normalized and already split.
    """
    ref_keywords_index = [1, 3, 4]
    assert isinstance(reference, list), "reference must be a list!"
    assert isinstance(transcript, list), "string must be a list!"

    assert len(reference) == 6 # grid samples are 6-words-long
    ref_align, trans_align = _needlemann_wunsch(reference=reference, transcript=transcript)
    ref_aligned_indices = [ref_align.index(reference[i]) for i in ref_keywords_index]


    trans_keywords = [trans_align[i] for i in ref_aligned_indices]
    assert len(trans_keywords) <= 3
    if return_idx:
        trans_keywords_indices = find_ordered_indices(transcript, trans_keywords)
        if l:=trans_keywords_indices[len(trans_keywords_indices)-1]:
            assert l < len(transcript)# last index cannot be greater than length of original string
        return trans_keywords_indices

    return trans_keywords


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
    name = f"the Needleman-Wunsch-WER of human results and {config.model}({config.model_type})"
    r_val, p_val = plot_regr_line_for_spearman_corr(df[["wers_human_kw", "wers_machine_kw"]],
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

    name = f"the Needleman-Wunsch-WER of human results and average log probability score of {config.model}({config.model_type})"
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

        plot_needleman_wunsch_wer_to_snr(df=df_single_run[["human_transcripts_kw", "machine_transcripts_kw", "snr", "references_kw", "model_type"]],
                                         shifting_attribute="model_type",
                                         output_path=config.output_path)

        corr_summary = plot_regr_lines(df_single_run, config)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine_kw", "model_type", "listener"]],
                                  correlate_to="wers_machine_kw",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine_kw", "model_type", "listener"]],
                                  correlate_to="wers_machine_kw",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "avg_logprobs", "model_type", "listener"]],
                                  correlate_to="avg_logprobs",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path)


        if config.extract_logprobs:
            keywords_index = [1, 3, 4]
            entropies = []
            tokenizer = get_tokenizer(multilingual=False)
            counter = 0

            error_counter = 0

            for index, row in df_single_run.iterrows():
                logprob_tensor = torch.load(Path.cwd()/row["logprobs_paths"])
                tokens = row["tokens"]
                decoded_tokens = [tokenizer.decode_with_timestamps([t]) for t in tokens]
                if len(decoded_tokens) != logprob_tensor.shape[0]:
                    #print(f"{counter = }, ({len(decoded_tokens) = } != {logprob_tensor.shape[0] = })")
                    error_counter+=1
                    entropies.append([np.nan, np.nan, np.nan])
                    continue
                idx_to_keep = ["<|" not in t and "|>" not in t for t in decoded_tokens] # rm timestamp tokens

                logprob_tensor = logprob_tensor[idx_to_keep]
                decoded_tokens_without_timestamp_tokens = [t for t,b in zip(decoded_tokens, idx_to_keep) if b] # todo make a copy of this

                # normalize for alignment
                decoded_tokens_without_timestamp_tokens = [o.lower().strip() for o in decoded_tokens_without_timestamp_tokens]
                decoded_tokens_without_timestamp_tokens = normalize(decoded_tokens_without_timestamp_tokens, apply_separate_numbers_from_letter=False, apply_werpy_normalize=False)
                #get keywords
                ref = row["references"]
                trans_keywords_indices = get_only_keywords_using_alignments(ref.split(), decoded_tokens_without_timestamp_tokens, return_idx=True)
                assert len(trans_keywords_indices) == 3
                logprob_tensor = np.array(logprob_tensor.cpu())

                e_arr = []
                for idx in trans_keywords_indices:
                    if idx is None:
                        e_arr.append(np.nan)
                        continue
                    e = entropy(np.exp(logprob_tensor[idx]))
                    e_arr.append(e)
                counter+=1
                entropies.append(e_arr)


            print(f"{error_counter = }")
            df_single_run["avg_entropy"] = entropies

            plot_entropy(df_single_run,
                shifting_attribute="model_type",
                output_path=config.output_path)


            #dont touch this
            boxplot_corr_per_listener(
                df_single_run[["wers_human_kw", "avg_entropy", "model_type", "listener"]],
                correlate_to="avg_entropy",
                model=config.model,
                model_type=config.model_type,
                output_path=config.output_path)

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
             extract_logprobs: bool,
             device: torch.device) -> Tuple[List[dict], pd.DataFrame]:

    data_path = output_path / "data"

    avg_logprobs = []
    references = []
    machine_transcripts = []
    tokens = []
    human_transcripts_kw = []
    snr = []
    listener = []
    audio_paths = []
    logprobs_paths = []

    counter = 0

    for file in data_path.iterdir():
        if counter == 2000:
            pass
        counter += 1
        with open(file) as f:
            json_file = json.load(f)

            if json_file["prediction_result"]["text"] == "" : #nothing recognized!
                avg_logprobs.append(torch.nan)
                machine_transcripts.append("")
                tokens.append([])
            else:
                avg_logprobs.append(np.mean([float(segment["avg_logprob"]) for segment in json_file["prediction_result"]["segments"]]))

                machine_transcripts.append(json_file["prediction_result"]["text"])

                tmp = []
                for segment in json_file["prediction_result"]["segments"]:
                    tmp.extend(segment["tokens"])
                tokens.append(tmp)

            references.append(json_file["sentence"])
            audio_paths.append(json_file["audio_path"])

            if dataset_type != "grid":
                human_transcripts_kw.append(json_file["human_recognized_words"])
                snr.append(int(json_file["snr_db"]))
                listener.append(json_file["listener"])

            if extract_logprobs:
                logprobs_paths.append(json_file["prediction_result"]["logprobs_path"])

    machine_transcripts = normalize(machine_transcripts)
    machine_transcripts_kw = [get_only_keywords_using_alignments(reference=r.split(), transcript=t.split()) for r, t in zip(references, machine_transcripts)]
    machine_transcripts_kw = [" ".join(w for w in t if w) for t in machine_transcripts_kw]
    human_transcripts_kw = normalize(human_transcripts_kw)

    references_kw = [get_only_keywords(o) for o in references]

    wers_machine = wer_needleman_wunsch_per_sample(references=references, transcripts=machine_transcripts)
    wers_machine_kw = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=machine_transcripts_kw)


    avg_logprobs = avg_logprobs
    if dataset_type != "grid":
        wers_human_kw = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=human_transcripts_kw)

    summary = []
    metric_names = ["Logprob(per sequence)", "WER (machine)", "WER (machine, kw only)"]
    metrics = [avg_logprobs, wers_machine, wers_machine_kw]
    if dataset_type != "grid":
        metric_names.append("WER (human study, kw only)")
        metrics.append(wers_human_kw)

    for n, m in zip(metric_names, metrics):
        summary.append({
            "metric_name": n,
            "mean": np.mean(m),
            "median": np.median(m),
            "std": np.std(m),
            "n": len(m)
        })

    data = {
        "avg_logprobs": avg_logprobs,
        "references": references,
        "references_kw": references_kw,
        "wers_machine": wers_machine,
        "wers_machine_kw": wers_machine_kw,
        "machine_transcripts": machine_transcripts,
        "tokens": tokens,
        "machine_transcripts_kw": machine_transcripts_kw,
        "audio_paths": audio_paths,
    }

    if dataset_type != "grid":
        data.update({
        "wers_human_kw": wers_human_kw,
        "human_transcripts_kw": human_transcripts_kw,
        "listener": listener,
        "snr": snr,
        })

    if extract_logprobs:
        data.update({
            "logprobs_paths": logprobs_paths,
        })

    df = pd.DataFrame(data)
    return summary, df
