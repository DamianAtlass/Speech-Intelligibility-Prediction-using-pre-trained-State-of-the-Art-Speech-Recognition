import json
from typing import Tuple, List, Literal, cast
import pandas as pd
import torch
from pandas import DataFrame
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
    boxplot_microscopic_corr_per_listener, join_kw_list_if_necessary
from utils.werpy_utils import normalize
from utils.wer_needleman_wunsch import wer_needleman_wunsch, wer_needleman_wunsch_per_sample, _needlemann_wunsch
from utils.dataset_utils import get_dataset

logger = logging.getLogger(__name__)
from utils.new_config_dataclass import InferenceConfig

from panphon.distance import Distance
dist = Distance()

from phonemizer.backend import EspeakBackend
backend = EspeakBackend("en-us")
phonemize = backend.phonemize

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

def get_kw_by_index(string) -> str:
    """
    Return only the words at the keyword indices. Only use when you expect the correct length of 6 words.
    """
    string = string.split()
    if len(string) != 6:
        raise ValueError(f"Expected 6 words, got {len(string)}")

    keywords_index = [1, 3, 4]

    string = [s for i,s in enumerate(string) if i in keywords_index]
    return " ".join(string)

def get_kw_using_needle_man_wunsch_alignments(
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

def get_kw_by_identity(
        reference_kw: list[str],
        transcript: list[str],
        return_idx=False) -> list[str|None]|list[int|None]:
    r = []
    for kw in reference_kw:
        if kw in transcript:
            r.append(transcript.index(kw) if return_idx else kw)
        else:
            r.append(None)

    rest = len(reference_kw) - len(r)
    r.extend([None for _ in range(rest)])
    return r

def get_kw_using_mixed_approaches(
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
                kw_phonemized = phonemize([kw])[0]
                phonetic_transcript = phonemize(transcript)
                distance = [dist.feature_edit_distance(source=kw_phonemized, target=o) for o in phonetic_transcript]

                idx = distance.index(min(distance))
                keywords_or_indices.append(idx if return_idx else transcript[idx])

    if return_idx:
        rest = len(reference_kw) - len(keywords_or_indices)
        keywords_or_indices.extend([None for _ in range(rest)])
    return keywords_or_indices

def get_kw_using_phonetic_similarity(
    reference_kw: list[str],
    transcript: list[str],
    return_idx=False) -> list[str]|list[int|None]:
    error_threshold = float("inf")

    r = []
    for kw in reference_kw:
        if kw in transcript:
            r.append(transcript.index(kw) if return_idx else kw)
        else:
            kw_phonemized = phonemize([kw])[0]
            phonetic_transcript = phonemize(transcript)
            distance = [dist.feature_edit_distance(source=kw_phonemized, target=o) for o in phonetic_transcript]

            if min(distance) <= error_threshold:
                idx = distance.index(min(distance))
                r.append(idx if return_idx else transcript[idx])
            else:
                raise RuntimeError("Should never reach here if error_threshold == inf")
                r.append(None)

    if return_idx:
        rest = len(reference_kw) - len(r)
        r.extend([None for _ in range(rest)])
    return r


def get_kw_by_accepting_other_options_from_vocab(
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

def get_kw_idx_through_time_alignments(reference_alignments: list[dict],
                                       transcript_alignments: list[dict]
                                       ) -> tuple[list[list[int]|None], list[int|None]]:
    grid_kw_index = [1,3,4]
    for r in reference_alignments:
        r["start"] = int(r["start"])/16000
        r["end"] = int(r["end"])/16000

    token_idx_list = []
    c = 0
    # for extracting logprobs later, it's needed to remember that words can have multiple tokens
    for t in transcript_alignments:
        tmp = []
        for _ in t["tokens"]:
            tmp.append(c)
            c+=1
        token_idx_list.append(tmp)

    reference_alignments = [a for a in reference_alignments if a["word"] not in ["sil", "sp"]]
    offset = reference_alignments[0]["start"]
    assert len(reference_alignments) == 6


    kw_token_idx_list: list[list[int]|None] = []
    kw_word_idx_list: list[int|None] = []
    for i in grid_kw_index:
        window_start = reference_alignments[i]["start"]
        window_end = reference_alignments[i]["end"]

        durations = []

        for word, token_indexes in zip(transcript_alignments, token_idx_list):
            s, e = float(word["start"])+offset, float(word["end"])+offset
            duration_in_window = max(0, min(window_end, e) - max(window_start, s))
            durations.append(duration_in_window)

        word_longest_in_window_idx = durations.index(max(durations)) if len(durations) > 0 else None
        kw_token_idx_list.append(token_idx_list[word_longest_in_window_idx] if word_longest_in_window_idx is not None else None)
        kw_word_idx_list.append(word_longest_in_window_idx)
    assert len(kw_token_idx_list) == 3


    return kw_token_idx_list, kw_word_idx_list

def plot_regr_lines(df: pd.DataFrame, config: InferenceConfig):
    summary = []
    #wers_human_kw, wers_machine_kw, avg_logprobs
    # calc person correlation

    # name = f"WER of human result and {config.model.name}({config.model.model_type})"
    # r_val, p_val, x_normality_p, y_normality_p = calc_pearson_corr(df[["wers_human_kw", "wers_machine_kw"]],
    #                                                                   output_path=config.output_path,
    #                                                                   name=name,
    #                                                                   xlabel=f"WER of human results (only keywords)",
    #                                                                   ylabel=f"WER ({config.model.name}, only keywords)")
    # summary.append({
    #     "metric": "person correlation",
    #     "of": name,
    #     "correlation_coefficient": f"{r_val:.10f}",
    #     "p_value": f"{p_val:.10f}",
    #     "x_normality_p_value": f"{x_normality_p:.10f}",
    #     "y_normality_p_value": f"{y_normality_p:.10f}",
    # })

    # name = f"the WER of human results and average log probability score of {config.model.name}({config.model.model_type})"
    # r_val, p_val, x_normality_p, y_normality_p = calc_pearson_corr(df[["wers_human_kw", "avg_logprobs"]],
    #                                                                   output_path=config.output_path,
    #                                                                   name=name,
    #                                                                   xlabel=f"WER of human results (only keywords)",
    #                                                                   ylabel=f"average logprob ({config.model.name})")
    # summary.append({
    #     "metric": "person correlation",
    #     "of": name,
    #     "correlation_coefficient": f"{r_val:.10f}",
    #     "p_value": f"{p_val:.10f}",
    #     "x_normality_p_value": f"{x_normality_p:.10f}",
    #     "y_normality_p_value": f"{y_normality_p:.10f}",
    # })

    # calc spearman correlation
    name = f"the WER of human transcripts and {config.model.name}({config.model.model_type}) whisper transcripts (kw only)"
    r_val, p_val = plot_regr_line_for_spearman_corr(df[["wers_human_kw", "wers_machine_kw"]],
                                                    output_path=config.output_path,
                                                    name=name,
                                                    xlabel=f"WER of human results (only keywords)",
                                                    ylabel=f"WER ({config.model.name}, only keywords)")

    summary.append({
        "metric": "spearman correlation",
        "of": name,
        "correlation_coefficient": f"{r_val:.10f}",
        "p_value": f"{p_val:.10f}",
    })

    name = f"the WER of human results and average log probability score of {config.model.name}({config.model.model_type})"
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
                            df_single_run: pd.DataFrame,
                            device: torch.device):
    # check for missing rows
    d = get_dataset(config.data.val_split)
    if len(d)!=len(df_single_run):
        print(f"Dataframe is missing rows! Expected {len(d["val"])}, got {len(df_single_run)}.")
    del d
    summary = get_summary(df=df_single_run, dataset_type=config.data.val_split.dataset_type)

    metrics = ["avg_logprobs", "wers_machine", "wers_machine_kw", "machine_transcripts_len"]

    corr_summary = None

    for m in tqdm(metrics):
        plot_metrics(data=[df_single_run[m]],
                     x_label=[config.model.model_type],
                     output_path=config.output_path)

    if config.data.val_split.dataset_type != "grid":
        plot_x_to_snr(df=df_single_run[["machine_transcripts", "snr", "model_type"]],
                      plotting_attribute="empty transcripts",
                      shifting_attribute_label="whisper",
                      shifting_attribute="model_type",
                      output_path=config.output_path)

        plot_wer_to_snr(df=df_single_run[["human_transcripts_kw", "machine_transcripts", "snr", "references_kw", "references", "model_type"]],
                        ref_col="references",
                        trans_col="machine_transcripts",
                        shifting_attribute="model_type",
                        output_path=config.output_path)

        plot_wer_to_snr(df=df_single_run[["human_transcripts_kw", "machine_transcripts_kw", "snr", "references_kw", "model_type"]],
                        ref_col="references_kw",
                        trans_col="machine_transcripts_kw",
                        shifting_attribute="model_type",
                        output_path=config.output_path)

        print("Generate correlation plots")
        corr_summary = plot_regr_lines(df_single_run, config)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine_kw", "model_type", "listener"]],
                                  correlate_to="wers_machine_kw",
                                  model=config.model.name,
                                  model_type=config.model.model_type,
                                  output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "wers_machine", "model_type", "listener"]],
                                  correlate_to="wers_machine",
                                  model=config.model.name,
                                  model_type=config.model.model_type,
                                  output_path=config.output_path)

        boxplot_corr_per_listener(df_single_run[["wers_human_kw", "avg_logprobs", "model_type", "listener"]],
                                  correlate_to="avg_logprobs",
                                  model=config.model.name,
                                  model_type=config.model.model_type,
                                  output_path=config.output_path)


    if config.extract_logprobs:
        if config.data.val_split.dataset_type == "grid_bc":
            plot_x_to_snr(df = df_single_run[["average_macroscopic_entropy", "snr", "model_type", "wers_human_kw"]],
                          plotting_attribute="average_macroscopic_entropy",
                          shifting_attribute_label="whisper",
                          shifting_attribute="model_type",
                          output_path=config.output_path)

        if config.data.val_split.dataset_type == "grid_bc":
            boxplot_corr_per_listener(
                df_single_run[["average_macroscopic_entropy", "wers_human_kw", "model_type", "listener"]],
                correlate_to="average_macroscopic_entropy",
                model=config.model.name,
                model_type=config.model.model_type,
                output_path=config.output_path)


        if config.extract_logprobs and config.data.val_split.dataset_type == "grid_bc":
            plot_microscopic_entropy(df_single_run[["entropies_kw", "listener", "model_type", "snr"]],
                                     entropy_col="entropies_kw",
                                     shifting_attribute="model_type",
                                     output_path=config.output_path)

            boxplot_microscopic_corr_per_listener(
                df_single_run[["entropies_kw", "listener", "model_type", "references_kw","human_transcripts_kw"]],
                entropy_col="entropies_kw",
                col_compare_against_ref_kw="human_transcripts_kw",
                output_path=config.output_path)

            #calibration
            boxplot_microscopic_corr_per_listener(
                df_single_run[["entropies_kw", "listener", "model_type", "references_kw", "estimated_transcript_kw"]],
                entropy_col="entropies_kw",
                col_compare_against_ref_kw="estimated_transcript_kw",
                output_path=config.output_path)

        # time alignments stuff
        if config.word_timestamps and config.data.val_split.dataset_type == "grid_bc":
            time_align_folder = config.output_path /"plots_from_time_alignments"
            time_align_folder.mkdir(parents=True, exist_ok=True)
            plot_wer_to_snr(
                df=df_single_run[
                ["human_transcripts_kw", "machine_trans_kw_from_time_align", "snr", "references_kw", "model_type"]],
                ref_col="references_kw",
                trans_col="machine_trans_kw_from_time_align",
                shifting_attribute="model_type",
                output_path=time_align_folder)


            if config.extract_logprobs:

                plot_microscopic_entropy(df_single_run[["entropies_kw_from_time_align", "listener", "model_type", "snr"]],
                                         entropy_col="entropies_kw_from_time_align",
                                         shifting_attribute="model_type",
                                         output_path=time_align_folder)

                boxplot_microscopic_corr_per_listener(
                    df_single_run[["references_kw", "listener", "model_type", "entropies_kw_from_time_align", "human_transcripts_kw"]],
                    entropy_col="entropies_kw_from_time_align",
                    col_compare_against_ref_kw="human_transcripts_kw",
                    output_path=time_align_folder)


                boxplot_microscopic_corr_per_listener(
                    df_single_run[["references_kw", "listener", "model_type", "entropies_kw_from_time_align", "machine_trans_kw_from_time_align"]],
                    entropy_col="entropies_kw_from_time_align",
                    col_compare_against_ref_kw="machine_trans_kw_from_time_align",
                    output_path=time_align_folder)

            idx_list = [[], [], []]
            foo = []
            for _, row in tqdm(df_single_run.iterrows(), total=len(df_single_run)):
                indexes = row["trans_kw_idx_from_align"]
                if len(indexes) > 0:
                    ts = foo.append(row["transcript_alignments"][0]["start"])
                    if ts != 0 and ts is not None:
                        pass
                for i, kw_idx in enumerate([1,3,4]):
                    if indexes[i] is not None:
                        idx_list[i].append(indexes[i])

            idx_list = [np.array(a) for a in idx_list]
            summary.append({f"kw at index {i}": {"mean idx of word": arr.mean(), "std": arr.std()} for i, arr in zip( [1,3,4], idx_list)})




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
    metric_names = ["Logprob (per sequence)", "WER (machine)", "WER (machine, kw only)", "transcript length"]
    metrics_col = ["avg_logprobs", "wers_machine", "wers_machine_kw", "machine_transcripts_len"]

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

def get_data(model: str,
             output_path: Path,
             dataset_type: str,
             extract_logprobs: bool,
             word_timestamps: bool,
             device: torch.device) -> DataFrame:

    file_name_df = output_path/"df.pkl"
    if file_name_df.exists():
        print("Load df from disk.")
        df: pd.DataFrame = pd.read_pickle(file_name_df)
        return df
    else:
        if (output_path/"summary.json").exists():
            os.remove(output_path/"summary.json")

    if model == "whisper":
        return get_data_whisper(output_path, dataset_type, extract_logprobs, word_timestamps, device)
    elif model == "parakeet":
        return get_data_parakeet(output_path, dataset_type, extract_logprobs, word_timestamps, device)
    else:
        raise ValueError(f"Unknown model: {model}")

def get_data_whisper(output_path: Path,
                     dataset_type: str,
                     extract_logprobs: bool,
                     word_timestamps: bool,
                     device: torch.device) -> pd.DataFrame:

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
    json_path = []
    references_alignments = []
    transcript_alignments = []

    logger.info("Read files...")
    counter = 0
    # read files
    for file in tqdm(data_path.iterdir(), total=len(list(data_path.iterdir()))):
        if counter == 10000:
            pass
        counter += 1
        with open(file) as f:
            json_file = json.load(f)
            json_path.append(str(file.relative_to(Path.cwd())))

            if json_file["prediction_result"]["text"] == "" : #nothing recognized!
                avg_logprobs.append(torch.nan)
                machine_transcripts.append("")
                if extract_logprobs:
                    decoded_tokens_with_timestamps.append([])
                if word_timestamps:
                    transcript_alignments.append([])
            else:
                avg_logprobs.append(np.mean([float(segment["avg_logprob"]) for segment in json_file["prediction_result"]["segments"]]))

                machine_transcripts.append(json_file["prediction_result"]["text"])
                if extract_logprobs:
                    decoded_tokens_with_timestamps.append(json_file["prediction_result"]["decoded_tokens_with_timestamps"])

                words = []
                for s in json_file["prediction_result"]["segments"]:
                    words.extend(s["words"])
                for w in words:
                    w.pop("probability") #dont need that currently
                transcript_alignments.append(words)

            references.append(json_file["sentence"])
            references_alignments.append([{"start": a[0], "end": a[1], "word": a[2]} for a in json_file["alignment"]])
            audio_paths.append(json_file["audio_path"])

            if dataset_type != "grid":
                human_transcripts_kw.append(json_file["human_recognized_words"])
                snr.append(int(json_file["snr_db"]))
                listener.append(json_file["listener"])

            if extract_logprobs:
                logprobs_paths.append(json_file["prediction_result"]["logprobs_path"])
    #post-processing
    references_kw: list[str] = [get_kw_by_index(o) for o in references]
    machine_transcripts: list[str] = normalize(machine_transcripts)
    human_transcripts_kw: list[str] = normalize(human_transcripts_kw)

    machine_transcripts_kw: list[list[str]] = cast(
        list[list[str]],
        [get_kw_by_identity(
            reference_kw=r.split(),
            transcript=t.split())
            for r, t in zip(references_kw, machine_transcripts)]
    )

    wers_machine: list[float] = wer_needleman_wunsch_per_sample(references=references, transcripts=machine_transcripts)
    wers_machine_kw: list[float] = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=join_kw_list_if_necessary(machine_transcripts_kw))

    if dataset_type != "grid":
        wers_human_kw = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=human_transcripts_kw)

    data = {
        "avg_logprobs": avg_logprobs,
        "references": references,
        "references_alignments": references_alignments,
        "references_kw": references_kw,
        "wers_machine": wers_machine,
        "wers_machine_kw": wers_machine_kw,
        "machine_transcripts": machine_transcripts,
        "machine_transcripts_kw": machine_transcripts_kw,
        "audio_paths": audio_paths,
        "json_path": json_path
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
            "decoded_tokens_with_timestamps": decoded_tokens_with_timestamps,
            "logprobs_paths": logprobs_paths,
        })
    if word_timestamps:
        data.update({
            "transcript_alignments": transcript_alignments
        })

    df = pd.DataFrame(data)

    # evaluate logprobs
    if extract_logprobs:
        print("Evaluate logprobs")
        # for logprobs
        entropies_kw: list[list[float|np.nan]] = []
        average_macroscopic_entropy = []
        estimated_transcript_keywords_indices: list[list[int|None]] = []
        estimated_transcript_keywords: list[list[str|None]] = []
        #for time_alignments
        machine_trans_kw_from_time_align: list[list[str|None]] = []
        trans_kw_idx_from_align: list[list[int|None]] = []
        entropies_kw_from_time_align: list[list[float|np.nan]] = []

        counter = 0
        no_transcript_counter = 0

        for index, row in tqdm(df.iterrows(), total=len(df)):
            #load logprobs
            if no_transcript:=(row["logprobs_paths"] == ""):
                no_transcript_counter += 1
            if not no_transcript:
                logprob_path = Path.cwd() / "inferences" / output_path / "logprobs" / Path(
                    row["logprobs_paths"]).name
                logprob_tensor = torch.load(logprob_path)

                # calculate entropy
                posteriors = logprob_tensor.exp()
                del logprob_tensor
                decoded_tokens_with_timestamps = row["decoded_tokens_with_timestamps"]
                assert len(decoded_tokens_with_timestamps) == posteriors.shape[0]
                assert torch.round(posteriors.sum(), decimals=2).item() == len(decoded_tokens_with_timestamps)
                entropies_per_token = Categorical(probs=posteriors).entropy().to(device)
                del posteriors
                assert len(entropies_per_token) == len(decoded_tokens_with_timestamps)

                ## rm timestamp tokens
                no_timestamp_idx = ["<|" not in t and "|>" not in t for t in
                               decoded_tokens_with_timestamps]
                entropies_per_token = entropies_per_token[no_timestamp_idx]
                decoded_tokens_without_timestamp_tokens = [t for t, b in zip(decoded_tokens_with_timestamps, no_timestamp_idx) if b]
                del decoded_tokens_with_timestamps

                average_macroscopic_entropy.append(float(entropies_per_token.mean()))

                ## 1) get kw idx by: get_only_keywords_with_different_approaches
                ### find "correct" kw position
                decoded_tokens_without_timestamp_tokens: list[str] = [o.lower().strip() for o in
                                                           decoded_tokens_without_timestamp_tokens]
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
                estimated_transcript_kw_idx = cast(list[int | None], get_kw_using_mixed_approaches(
                     reference_kw=row["references_kw"].split(),
                     transcript=decoded_tokens_without_timestamp_tokens,
                     return_idx=True))
                assert len(estimated_transcript_kw_idx) == 3
                #technically, this is not taking into account, words could be split up into subwords, which is however not likely considering the grid vocab
                estimated_transcript_kw: list[str|None] = [None if idx is None else decoded_tokens_without_timestamp_tokens[idx] for idx in estimated_transcript_kw_idx]

                estimated_transcript_keywords_indices.append(estimated_transcript_kw_idx)
                estimated_transcript_keywords.append(estimated_transcript_kw)

                tmp_kw_entropy: list[float|np.nan] = []
                for idx in estimated_transcript_kw_idx:
                    tmp_kw_entropy.append(np.nan if idx is None else float(entropies_per_token[idx]))
                entropies_kw.append(tmp_kw_entropy)

                ## 2) get kw idx by using the time-alignments
                if word_timestamps:
                    assert sum([len(a["tokens"]) for a in row["transcript_alignments"]]) == len(
                        decoded_tokens_without_timestamp_tokens)
                    place_holder = get_kw_idx_through_time_alignments(
                        row["references_alignments"], row["transcript_alignments"])
                    kw_token_idx_from_alignment: list[list[int | None]] = place_holder[0]
                    kw_word_idx_list: list[int | None] = place_holder[1]


                    trans_kw_idx_from_align.append(kw_word_idx_list)
                    kw_from_alignment: list[str|None] = [(row["transcript_alignments"][idx]["word"] if idx is not None else None) for idx in kw_word_idx_list]
                    kw_from_alignment: list[str|None] = [o.lower().strip() for o in kw_from_alignment]
                    machine_trans_kw_from_time_align.append(kw_from_alignment)

                    # assume word_timestamps and extract_logprobs are True
                    # kw_token_idx_from_alignment: list[list[int|None]] idx for logprobs
                    tmp_kw_entropy: list[float|np.nan] = []
                    for idx in kw_token_idx_from_alignment:
                        tmp_kw_entropy.append(np.nan if idx is None else float(entropies_per_token[idx].mean()))
                    assert all([len(a)==1 for a in kw_token_idx_from_alignment if a is not None])
                    entropies_kw_from_time_align.append(tmp_kw_entropy)

            else:
                # no transcript

                average_macroscopic_entropy.append(torch.nan)
                estimated_transcript_keywords_indices.append([None, None, None])
                estimated_transcript_keywords.append([None, None, None])
                entropies_kw.append([torch.nan, torch.nan, torch.nan])

                if word_timestamps:
                    machine_trans_kw_from_time_align.append([None, None, None])
                    entropies_kw_from_time_align.append([torch.nan, torch.nan, torch.nan])
                    trans_kw_idx_from_align.append([None, None, None])


        df["average_macroscopic_entropy"] = average_macroscopic_entropy
        df["estimated_transcript_kw_idx"] = estimated_transcript_keywords_indices
        df["estimated_transcript_kw"] = estimated_transcript_keywords
        df["entropies_kw"] = entropies_kw
        del (average_macroscopic_entropy, estimated_transcript_keywords_indices, estimated_transcript_keywords, entropies_kw)

        if word_timestamps:
            df["machine_trans_kw_from_time_align"] = machine_trans_kw_from_time_align
            df["entropies_kw_from_time_align"] = entropies_kw_from_time_align
            df["trans_kw_idx_from_align"] = trans_kw_idx_from_align




    df.to_pickle(output_path/"df.pkl")
    return df


def get_data_parakeet(output_path: Path,
                     dataset_type: str,
                     extract_logprobs: bool,
                     device: torch.device) -> pd.DataFrame:
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

            if json_file["prediction_result"]["text"] == "":  # nothing recognized!
                avg_logprobs.append(torch.nan)
                machine_transcripts.append("")
                if extract_logprobs:
                    decoded_tokens_with_timestamps.append([])
            else:
                y_seq = json_file["prediction_result"]["y_sequence"]
                avg = json_file["prediction_result"]["score"] / (len(y_seq) - y_seq.count(1024))  # 1024 is mask token
                avg_logprobs.append(avg)

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

    references_kw = [get_kw_by_index(o) for o in references]
    machine_transcripts = normalize(machine_transcripts)
    # machine_transcripts_kw = [get_only_keywords_using_alignments(reference=r.split(), transcript=t.split()) for r, t in zip(references, machine_transcripts)]
    machine_transcripts_kw = [get_kw_by_identity(reference_kw=r.split(), transcript=t.split()) for r, t in
                              zip(references_kw, machine_transcripts)] # out of date now

    recognize_kw = np.sum([np.sum([1 for _ in keywords]) for keywords in machine_transcripts_kw])
    recognize_kw_percent = (recognize_kw / (len(machine_transcripts_kw) * 3))
    print(f"recognized keywords: {round(recognize_kw_percent, 2) * 100}%")
    machine_transcripts_kw = [" ".join(w for w in t if w) for t in machine_transcripts_kw] # outdated
    human_transcripts_kw = normalize(human_transcripts_kw)

    wers_machine = wer_needleman_wunsch_per_sample(references=references, transcripts=machine_transcripts)
    wers_machine_kw = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=machine_transcripts_kw)

    if dataset_type != "grid":
        wers_human_kw = wer_needleman_wunsch_per_sample(references=references_kw, transcripts=human_transcripts_kw)

    data = {
        "avg_logprobs": avg_logprobs,
        "references": references,
        "references_kw": references_kw,
        "wers_machine": wers_machine,
        "wers_machine_kw": wers_machine_kw,
        "machine_transcripts": machine_transcripts,
        #"decoded_tokens_with_timestamps": decoded_tokens_with_timestamps,
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

    # evaluate logprobs
    found_kw = 0
    no_kw_in_sentence_found = 0
    # todo fix logprob extraction with parakeet!!!
    if extract_logprobs:
        print("Evaluate logprobs")
        entropies_kw = []
        average_macroscopic_entropy = []
        estimated_transcript_keywords_indices = []

        counter = 0
        error_counter = 0

        for index, row in tqdm(df.iterrows(), total=len(df)):
            # load logprobs
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
            entropies_per_token = Categorical(probs=posteriors).entropy().to(device)
            del posteriors
            assert len(entropies_per_token) == len(decoded_tokens_with_timestamps)

            # rm timestamp tokens
            no_timestamp_idx = ["<|" not in t and "|>" not in t for t in
                                decoded_tokens_with_timestamps]
            entropies_per_token = entropies_per_token[no_timestamp_idx]
            decoded_tokens_without_timestamp_tokens = [t for t, b in
                                                       zip(decoded_tokens_with_timestamps, no_timestamp_idx) if
                                                       b]
            del decoded_tokens_with_timestamps

            average_macroscopic_entropy.append(float(entropies_per_token.mean()))

            # get kw specific entropy
            decoded_tokens_without_timestamp_tokens = [o.lower().strip() for o in
                                                       decoded_tokens_without_timestamp_tokens]
            ## find "correct" kw position
            decoded_tokens_without_timestamp_tokens = normalize(decoded_tokens_without_timestamp_tokens,
                                                                apply_separate_numbers_from_letter=False,
                                                                apply_numbers_to_words=True,
                                                                apply_werpy_normalize=False)

            # trans_keywords_indices = get_only_keywords_using_alignments(ref.split(), decoded_tokens_without_timestamp_tokens, return_idx=True)
            # trans_keywords_indices = get_only_keywords_by_identity(row["references_kw"].split(),
            #                                                       decoded_tokens_without_timestamp_tokens,
            #                                                       return_idx=True)
            # trans_keywords_indices = get_only_keywords_by_accepting_other_options(row["references_kw"].split(),
            #                                                                      decoded_tokens_without_timestamp_tokens,
            #                                                                      return_idx=True)
            # transcript_keywords_indices: list[int|None] = get_only_keywords_by_phonetic_similarity(reference_kw=row["references_kw"].split(),
            #                                                                   transcript=decoded_tokens_without_timestamp_tokens,
            #                                                                   return_idx=True)
            transcript_keywords_indices: list[int | None] = get_kw_using_mixed_approaches(
                reference_kw=row["references_kw"].split(),
                transcript=decoded_tokens_without_timestamp_tokens,
                return_idx=True)
            # transcript_keywords_indices = [1,3,4]
            estimated_transcript_keywords_indices.append(transcript_keywords_indices)
            assert len(transcript_keywords_indices) == 3

            tmp_found_kw = np.sum([1 for o in transcript_keywords_indices if o is not None])
            found_kw += tmp_found_kw
            no_kw_in_sentence_found += tmp_found_kw == 0

            tmp_wk_entropy = []
            for idx in transcript_keywords_indices:
                tmp_wk_entropy.append(np.nan if idx is None else float(entropies_per_token[idx]))
            del entropies_per_token
            counter += 1
            entropies_kw.append(tmp_wk_entropy)

        df["average_macroscopic_entropy"] = average_macroscopic_entropy
        df["estimated_transcript_kw_idx"] = estimated_transcript_keywords_indices
        df["entropies_kw"] = entropies_kw

        print(f"{error_counter = }")
        print(f"{found_kw = }, -> {round(found_kw / (((len(df) - error_counter) * 3)), 2) * 100}%")
        print(
            f"{no_kw_in_sentence_found = }, -> {round(no_kw_in_sentence_found / len(df) - error_counter, 2) * 100}%")

    df.to_pickle(output_path / "df.pkl")
    return df
