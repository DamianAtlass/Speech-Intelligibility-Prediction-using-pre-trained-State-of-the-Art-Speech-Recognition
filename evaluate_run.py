from utils.config_dataclasses import get_config, unfold_config
from pathlib import Path
import json
from utils.logging_utils import catch_time

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
from utils.cuda_utils import select_device
from utils.werpy_utils import additional_normalization, calculate_wers_with_norm
import logging
logger = logging.getLogger(__name__)
from utils.evaluate_utils import get_only_keywords, plot_metrics, plot_correlations, plot_srt
import pandas as pd
from typing import Tuple, List


def get_data(output_path: Path, dataset_type: str, device: torch.device) -> Tuple[List[dict], pd.DataFrame]:
    if not device:
        device = select_device()

    data_path = output_path / "data"

    avg_logprobs = []
    references = []
    machine_transcripts = []
    human_transcripts_kw = []
    snr = []

    for file in data_path.iterdir():
        with open(file) as f:
            json_file = json.load(f)

            if json_file["prediction_result"]["text"] == "" : #nothing recognized!
                avg_logprobs.append(torch.nan)
                machine_transcripts.append("")
            else:
                avg_logprobs.append(json_file["prediction_result"]["segments"][0]["avg_logprob"])
                machine_transcripts.append(json_file["prediction_result"]["text"])

            human_transcripts_kw.append(json_file["human_recognized_words"])
            references.append(json_file["sentence"])
            snr.append(int(json_file["snr_db"]))


    machine_transcripts = additional_normalization(machine_transcripts)
    machine_transcripts_kw = [get_only_keywords(o) for o in machine_transcripts]

    human_transcripts_kw = additional_normalization(human_transcripts_kw)

    references_kw = [get_only_keywords(o) for o in references]

    wers_machine = calculate_wers_with_norm(reference=references, hypothesis=machine_transcripts).to(device)
    wers_machine_kw = calculate_wers_with_norm(reference=references_kw, hypothesis=machine_transcripts_kw).to(device)
    wers_human_kw = calculate_wers_with_norm(reference=references_kw, hypothesis=human_transcripts_kw).to(device)

    avg_logprobs = torch.Tensor(avg_logprobs).to(device)

    summary = []
    for name, values in zip(["Logprob(per sequence)", "WER (machine)", "WER (machine, kw only)", "WER (human study, kw only)"],
                            [avg_logprobs, wers_machine, wers_machine_kw, wers_human_kw]):
        summary.append({
            "metric_name": name,
            "mean": values.mean().item(),
            "median": values.median().item(),
            "std": values.std().item(),
            "n": values.shape[0]
        })

    df = pd.DataFrame({
        "snr": snr,
        "avg_logprobs": avg_logprobs.cpu(),
        "references": references,
        "references_kw": references_kw,
        "wers_machine": wers_machine.cpu(),
        "wers_machine_kw": wers_machine_kw.cpu(),
        "machine_transcripts": machine_transcripts,
        "machine_transcripts_kw": machine_transcripts_kw,
        "wers_human_kw": wers_human_kw.cpu(),
        "human_transcripts_kw": human_transcripts_kw,
    })
    return summary, df

def evaluate_run(path: Path, device: torch.device | None = None):

    config = get_config(path / "config.ini")
    unfolded_configs = unfold_config(config)

    if len(unfolded_configs)==1:
        with catch_time() as t:
            summary, df = get_data(config.output_path, config.dataset_type)
        print(f"Reading the generated files took: {t():.4f} s")

        for s, metric in zip(summary, [df["avg_logprobs"], df["wers_machine"], df["wers_machine_kw"], df["wers_human_kw"]]):
            plot_metrics([metric],
                         f"Average {s["metric_name"]}s for {config.model}({config.model_type})",
                         s["metric_name"],
                         [config.model_type],
                         config.output_path)

        summary2 = plot_correlations(df, config)

        with open(config.output_path / "summary.json", 'w') as f:
            json.dump({"summary1:": summary, "correlation:": summary2}, f, indent=4)

        plot_srt(df[["wers_human_kw", "wers_machine_kw", "snr"]], config)


    else:
        summary_arr, avg_logprobs_arr, wers_machine_arr, wers_machine_kw_arr, wers_human_kw_arr = [], [], [], [], []
        list_field = config.get_list_fields()[0]

        for c in unfolded_configs:
            with catch_time() as t:
                #avg_logprobs, wers_machine, wers_machine_kw, wers_human_kw
                summary, df = get_data(c.output_path, c.dataset_type, device=device)
            print(f"Reading the generated files took: {t():.4f} s")

            for s, metric in zip(summary, [df["avg_logprobs"], df["wers_machine"], df["wers_machine_kw"], df["wers_human_kw"]]):
                plot_metrics([metric],
                            f"Average {s["metric_name"]}s",
                             s["metric_name"],
                             [getattr(c, list_field)],
                             c.output_path)

            summary2 = plot_correlations(df, c)

            with open(c.output_path / "summary.json", 'w') as f:
                json.dump({"summary1:": summary, "correlation:": summary2}, f, indent=4)

            summary_arr.append(summary)
            avg_logprobs_arr.append(df["avg_logprobs"])
            wers_machine_arr.append(df["wers_machine"])
            wers_machine_kw_arr.append(df["wers_machine_kw"])
            wers_human_kw_arr.append(df["wers_human_kw"])

            plot_srt(df[["wers_human_kw", "wers_machine_kw", "snr", "machine_transcripts_kw", "human_transcripts_kw"]], c)

        # plot subplots
        for s_arr, metric_arr in zip(zip(*summary_arr), [avg_logprobs_arr, wers_machine_arr, wers_machine_kw_arr, wers_human_kw_arr]):
            plot_metrics(metric_arr,
                        f"Average {s_arr[0]['metric_name']}s",
                         s_arr[0]["metric_name"],
                         [c.model_type for c in unfolded_configs],
                         config.output_path)
    logger.info("Finished evaluation")


if __name__ == '__main__':
    evaluate_run(Path("inferences/tmp4"))