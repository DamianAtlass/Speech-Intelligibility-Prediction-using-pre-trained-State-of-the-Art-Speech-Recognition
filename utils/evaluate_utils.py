import json
from typing import Tuple, List

import pandas as pd
import torch
from torch.distributions import Categorical
from pathlib import Path
import werpy
import logging
from scipy.stats import entropy
import numpy as np
from tqdm import tqdm
import os

from utils.plotting_utils import plot_regr_line_for_spearman_corr, plot_metrics, \
    plot_wer_to_snr, boxplot_corr_per_listener, plot_microscopic_entropy, plot_x_to_snr, \
    boxplot_microscopic_corr_per_listener
from utils.werpy_utils import normalize
from utils.wer_needleman_wunsch import wer_needleman_wunsch, wer_needleman_wunsch_per_sample, _needlemann_wunsch

logger = logging.getLogger(__name__)
from utils.config_dataclasses import InferenceConfig

from phonemizer import phonemize
from panphon.distance import Distance
dist = Distance()

grid_vocab = {
    "color": ['blue', 'green', 'red', 'white'], #4 items, index 1
    "letter": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z'], # 25 items, index 3
    "digit": ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero'] # 10 items, index 4
}

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

def get_only_keywords_by_index(string) -> str:
    """
    Return only the words at the keyword indices. Only use when you expect the correct length of 6 words.
    """
    string = string.split()
    if len(string) != 6:
        raise ValueError(f"Expected 6 words, got {len(string)}")

    keywords_index = [1, 3, 4]

    string = [s for i,s in enumerate(string) if i in keywords_index]
    return " ".join(string)

def get_only_keywords_using_alignments(
        reference: list[str],
        transcript: list[str],
        return_idx=False) -> list[str] | list[int]:
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

def get_only_keywords_by_identity(
        reference_kw: list[str],
        transcript: list[str],
        return_idx=False) -> list[str]|list[int|None]:
    r = []
    for kw in reference_kw:
        if kw in transcript:
            r.append(transcript.index(kw) if return_idx else kw)
        else:
            if return_idx:
                r.append(None)

    if return_idx:
        rest = len(reference_kw) - len(r)
        r.extend([None for _ in range(rest)])
    return r

def get_only_keywords_with_different_approaches(
    reference_kw: list[str],
    transcript: list[str],
    return_idx=False) -> list[str]|list[int|None]:

    re = lambda x: transcript.index(x) if return_idx else x

    keywords_or_indices = []
    for kw, expected_kw_position, other_options in zip(reference_kw, [1,3,4], grid_vocab.values()):
        if kw in transcript:
            keywords_or_indices.append(re(kw))
        else:
            added = False
            #check if kw_varient at expected position
            if len(transcript)>=expected_kw_position+1 and (word:=transcript[expected_kw_position]) in other_options:
                keywords_or_indices.append(re(word))
                continue

            # search kw option in whole transcript
            for word in transcript:
                if word in other_options:
                    keywords_or_indices.append(re(word))
                    added = True
                    break
            if added:
                continue
            else:
                kw_phonemized = phonemize(kw, language="en-us", backend="espeak")
                phonetic_transcript = phonemize(transcript, language="en-us", backend="espeak")
                distance = [dist.feature_edit_distance(source=kw_phonemized, target=o) for o in phonetic_transcript]

                idx = distance.index(min(distance))
                keywords_or_indices.append(idx if return_idx else transcript[idx])

    if return_idx:
        rest = len(reference_kw) - len(keywords_or_indices)
        keywords_or_indices.extend([None for _ in range(rest)])
    return keywords_or_indices

def get_only_keywords_by_phonetic_similarity(
    reference_kw: list[str],
    transcript: list[str],
    return_idx=False) -> list[str]|list[int|None]:
    error_threshold = float("inf")

    r = []
    for kw in reference_kw:
        if kw in transcript:
            r.append(transcript.index(kw) if return_idx else kw)
        else:
            kw_phonemized = phonemize(kw, language="en-us", backend="espeak")
            phonetic_transcript = phonemize(transcript, language="en-us", backend="espeak")
            distance = [dist.feature_edit_distance(source=kw_phonemized, target=o) for o in phonetic_transcript]

            if min(distance) <= error_threshold:
                idx = distance.index(min(distance))
                r.append(idx if return_idx else transcript[idx])
            else:
                r.append(None)

    if return_idx:
        rest = len(reference_kw) - len(r)
        r.extend([None for _ in range(rest)])
    return r


def get_only_keywords_by_accepting_other_options(
        reference_kw: list[str],
        transcript: list[str],
        return_idx=False) -> list[str] | list[int | None]:
    r = []
    for kw, other_options in zip(reference_kw, grid_vocab.values()):
        if kw in transcript:
            r.append(transcript.index(kw) if return_idx else kw)
        else:
            added = False
            for o in other_options:
                if o in transcript:
                    r.append(transcript.index(o) if return_idx else o)
                    added = True
                    break
            if added:
                continue
            else:
                r.append(None)

    if return_idx:
        rest = len(reference_kw) - len(r)
        r.extend([None for _ in range(rest)])
    return r

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
    name = f"the WER of human transcripts and {config.model}({config.model_type}) whisper transcripts (kw only)"
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
        print("Generate correlation plots")
        metrics.append(df_single_run["wers_human_kw"])

        plot_wer_to_snr(df=df_single_run[["human_transcripts_kw", "machine_transcripts", "snr", "references_kw", "references", "model_type"]],
                        only_kw=False,
                        shifting_attribute="model_type",
                        output_path=config.output_path)

        plot_wer_to_snr(df=df_single_run[["human_transcripts_kw", "machine_transcripts_kw", "snr", "references_kw", "model_type"]],
                        only_kw=True,
                        shifting_attribute="model_type",
                        output_path=config.output_path)

        corr_summary = plot_regr_lines(df_single_run, config)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine_kw", "model_type", "listener"]],
                                  correlate_to="wers_machine_kw",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine", "model_type", "listener"]],
                                  correlate_to="wers_machine",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "avg_logprobs", "model_type", "listener"]],
                                  correlate_to="avg_logprobs",
                                  model=config.model,
                                  model_type=config.model_type,
                                  output_path=config.output_path)

    if config.extract_logprobs:

            plot_x_to_snr(df = df_single_run[["average_macroscopic_entropy", "snr", "model_type", "wers_human_kw"]],
                          plotting_attribute="average_macroscopic_entropy",
                          shifting_attribute_label="whisper",
                          shifting_attribute="model_type",
                          output_path=config.output_path
                          )

            boxplot_corr_per_listener(
                df_single_run[["average_macroscopic_entropy", "wers_human_kw", "model_type", "listener"]],
                correlate_to="average_macroscopic_entropy",
                model=config.model,
                model_type=config.model_type,
                output_path=config.output_path)



            plot_microscopic_entropy(df_single_run[["entropies_kw", "listener", "model_type", "snr"]],
                                     shifting_attribute="model_type",
                                     output_path=config.output_path)

            boxplot_microscopic_corr_per_listener(
                df_single_run[["entropies_kw", "listener", "model_type", "references_kw","human_transcripts_kw",
                               "estimated_transcript_keywords_indices", "machine_transcripts"]],
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

def get_summary(df: pd.DataFrame,
                dataset_type: str) -> list[dict]:
    """
    Args:
        df: pd.DataFrame
        dataset_type: str

    Returns:
        summary: list[dict], a summary of some df columns
    """

    summary = []
    df["machine_transcripts_len"] = df["machine_transcripts"].map(lambda x: len(x.split()))
    metric_names = ["Logprob(per sequence)", "WER (machine)", "WER (machine, kw only)", "transcript length"]
    metrics_col = ["avg_logprobs", "wers_machine", "wers_machine_kw", "machine_transcripts_len"]
    if dataset_type != "grid":
        metric_names.append("WER (human study, kw only)")
        metrics_col.append("wers_human_kw")

    for n, m in zip(metric_names, metrics_col):
        values = df[m]
        summary.append({
            "metric_name": n,
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "n": len(values)
        })
    return summary

def get_data(output_path: Path,
             dataset_type: str,
             extract_logprobs: bool,
             device: torch.device) -> Tuple[List[dict], pd.DataFrame]:

    file_name_df = output_path/"df.pkl"
    if file_name_df.exists():
        print("Load df from disk.")
        df: pd.DataFrame = pd.read_pickle(file_name_df)
        summary = get_summary(df=df, dataset_type=dataset_type)
        return summary, df
    else:
        if (output_path/"summary.json").exists():
            os.remove(output_path/"summary.json")

    data_path = output_path / "data"

    avg_logprobs = []
    references = []
    machine_transcripts = []
    decoded_tokens_with_timestamps = []
    human_transcripts_kw = []
    snr = []
    listener = []
    audio_paths = []
    logprobs_paths = []

    counter = 0

    for file in tqdm(data_path.iterdir()):
        if counter == 10000:
            pass
        counter += 1
        with open(file) as f:
            json_file = json.load(f)

            if json_file["prediction_result"]["text"] == "" : #nothing recognized!
                avg_logprobs.append(torch.nan)
                machine_transcripts.append("")
                if extract_logprobs:
                    decoded_tokens_with_timestamps.append([])
            else:
                avg_logprobs.append(np.mean([float(segment["avg_logprob"]) for segment in json_file["prediction_result"]["segments"]]))

                machine_transcripts.append(json_file["prediction_result"]["text"])
                if extract_logprobs:
                    decoded_tokens_with_timestamps.append(json_file["prediction_result"]["decoded_tokens_with_timestamps"])

            references.append(json_file["sentence"])
            audio_paths.append(json_file["audio_path"])

            if dataset_type != "grid":
                human_transcripts_kw.append(json_file["human_recognized_words"])
                snr.append(int(json_file["snr_db"]))
                listener.append(json_file["listener"])

            if extract_logprobs:
                logprobs_paths.append(json_file["prediction_result"]["logprobs_path"])

    references_kw = [get_only_keywords_by_index(o) for o in references]
    machine_transcripts = normalize(machine_transcripts)
    #machine_transcripts_kw = [get_only_keywords_using_alignments(reference=r.split(), transcript=t.split()) for r, t in zip(references, machine_transcripts)]
    machine_transcripts_kw = [get_only_keywords_by_identity(reference_kw=r.split(), transcript=t.split()) for r, t in zip(references_kw, machine_transcripts)]

    recognize_kw = np.sum([np.sum([1 for _ in keywords]) for keywords in machine_transcripts_kw])
    recognize_kw_percent = (recognize_kw/(len(machine_transcripts_kw)*3))
    print(f"recognized keywords: {round(recognize_kw_percent, 2)*100}%")
    machine_transcripts_kw = [" ".join(w for w in t if w) for t in machine_transcripts_kw]
    human_transcripts_kw = normalize(human_transcripts_kw)

    wers_machine = wer_needleman_wunsch_per_sample(references=references, transcripts=machine_transcripts)
    wers_machine_kw = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=machine_transcripts_kw)

    avg_logprobs = avg_logprobs
    if dataset_type != "grid":
        wers_human_kw = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=human_transcripts_kw)

    data = {
        "avg_logprobs": avg_logprobs,
        "references": references,
        "references_kw": references_kw,
        "wers_machine": wers_machine,
        "wers_machine_kw": wers_machine_kw,
        "machine_transcripts": machine_transcripts,
        "decoded_tokens_with_timestamps": decoded_tokens_with_timestamps,
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
    summary = get_summary(df=df, dataset_type=dataset_type)

    # evaluate logprobs
    found_kw = 0
    no_kw_in_sentence_found = 0

    if extract_logprobs:
        print("Evaluate logprobs")
        entropies_kw = []
        average_macroscopic_entropy = []
        estimated_transcript_keywords_indices = []

        counter = 0
        error_counter = 0

        for index, row in tqdm(df.iterrows(), total=len(df)):
            #load logprobs
            if row["logprobs_paths"] == "":
                error_counter += 1
                continue
            logprob_path = Path.cwd() / "inferences" / output_path / "logprobs" / Path(
                row["logprobs_paths"]).name
            logprob_tensor = torch.load(logprob_path)

            # calculate entropy
            posteriors = logprob_tensor.exp()
            del logprob_tensor
            decoded_tokens_with_timestamps = row["decoded_tokens_with_timestamps"]
            assert len(decoded_tokens_with_timestamps) == posteriors.shape[0]
            assert torch.round(posteriors.sum(), decimals=2).item() == len(decoded_tokens_with_timestamps)
            entropies_per_token = Categorical(probs=posteriors).entropy()
            del posteriors
            assert len(entropies_per_token) == len(decoded_tokens_with_timestamps)

            # rm timestamp tokens
            no_timestamp_idx = ["<|" not in t and "|>" not in t for t in
                           decoded_tokens_with_timestamps]
            entropies_per_token = entropies_per_token[no_timestamp_idx]
            decoded_tokens_without_timestamp_tokens = [t for t, b in zip(decoded_tokens_with_timestamps, no_timestamp_idx) if
                                                       b]
            del decoded_tokens_with_timestamps

            average_macroscopic_entropy.append(entropies_per_token.mean().item())

            # get kw specific entropy
            decoded_tokens_without_timestamp_tokens = [o.lower().strip() for o in
                                                       decoded_tokens_without_timestamp_tokens]
            ## find "correct" kw position
            decoded_tokens_without_timestamp_tokens = normalize(decoded_tokens_without_timestamp_tokens,
                                                                apply_separate_numbers_from_letter=False,
                                                                apply_numbers_to_words=True,
                                                                apply_werpy_normalize=False)

            # trans_keywords_indices = get_only_keywords_using_alignments(ref.split(), decoded_tokens_without_timestamp_tokens, return_idx=True)
            #trans_keywords_indices = get_only_keywords_by_identity(row["references_kw"].split(),
            #                                                       decoded_tokens_without_timestamp_tokens,
            #                                                       return_idx=True)
            #trans_keywords_indices = get_only_keywords_by_accepting_other_options(row["references_kw"].split(),
            #                                                                      decoded_tokens_without_timestamp_tokens,
            #                                                                      return_idx=True)
            # transcript_keywords_indices: list[int|None] = get_only_keywords_by_phonetic_similarity(reference_kw=row["references_kw"].split(),
            #                                                                   transcript=decoded_tokens_without_timestamp_tokens,
            #                                                                   return_idx=True)
            transcript_keywords_indices: list[int | None] = get_only_keywords_with_different_approaches(
                 reference_kw=row["references_kw"].split(),
                 transcript=decoded_tokens_without_timestamp_tokens,
                 return_idx=True)
            #transcript_keywords_indices = [1,3,4]
            estimated_transcript_keywords_indices.append(transcript_keywords_indices)
            assert len(transcript_keywords_indices) == 3

            tmp_found_kw = np.sum([1 for o in transcript_keywords_indices if o is not None])
            found_kw += tmp_found_kw
            no_kw_in_sentence_found += tmp_found_kw == 0

            tmp_wk_entropy = []
            for idx in transcript_keywords_indices:
                tmp_wk_entropy.append(np.nan if idx is None else entropies_per_token[idx])

            counter += 1
            entropies_kw.append(tmp_wk_entropy)

        df["average_macroscopic_entropy"] = average_macroscopic_entropy
        df["estimated_transcript_keywords_indices"] = estimated_transcript_keywords_indices
        df["entropies_kw"] = entropies_kw

        print(f"{error_counter = }")
        print(f"{found_kw = }, -> {round(found_kw / (((len(df) - error_counter) * 3)), 2) * 100}%")
        print(
            f"{no_kw_in_sentence_found = }, -> {round(no_kw_in_sentence_found / len(df) - error_counter, 2) * 100}%")

    df.to_pickle(output_path/file_name_df)
    return summary, df
