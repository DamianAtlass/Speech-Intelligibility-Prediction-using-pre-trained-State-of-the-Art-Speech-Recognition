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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from utils.evaluate_utils import calc_pearson_corr, calc_spearman_corr, get_only_keywords, plot_metrics, plot_correlations


def get_data(output_path: Path, dataset_type: str) -> tuple:
    data_path = output_path / "data"
    device = select_device()

    avg_logprobs = []
    references = []
    machine_transcripts = []
    human_transcripts_kw = []

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


    machine_transcripts = additional_normalization(machine_transcripts)
    machine_transcripts_kw = [get_only_keywords(o) for o in machine_transcripts]

    human_transcripts_kw = additional_normalization(human_transcripts_kw)

    references_kw = [get_only_keywords(o) for o in references]

    wers_machine = calculate_wers_with_norm(machine_transcripts, references).to(device)
    wers_machine_kw = calculate_wers_with_norm(machine_transcripts_kw, references_kw).to(device)
    wers_human_kw = calculate_wers_with_norm(human_transcripts_kw, references_kw).to(device)

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

    with open(output_path/"summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    return summary, avg_logprobs, wers_machine, wers_machine_kw, wers_human_kw

def evaluate_run(path: Path):

    config = get_config(path / "config.ini")

    unfolded_configs = unfold_config(config)

    if len(unfolded_configs)==1:
        with catch_time() as t:
            summary, avg_logprobs, wers_machine, wers_machine_kw, wers_human_kw = get_data(config.output_path, config.dataset_type)
        print(f"Reading the generated files took: {t():.4f} s")

        for s, metric in zip(summary, [avg_logprobs, wers_machine, wers_machine_kw, wers_human_kw]):
            plot_metrics([metric],
                         f"Average {s["metric_name"]}s for {config.model}({config.model_type})",
                         s["metric_name"],
                         [config.model_type],
                         config.output_path)

        plot_correlations(wers_human_kw, wers_machine_kw, avg_logprobs, config)

    else:
        summary_arr, avg_logprobs_arr, wers_machine_arr, wers_machine_kw_arr, wers_human_kw_arr = [], [], [], [], []
        list_field = config.get_list_fields()[0]

        for c in unfolded_configs:
            with catch_time() as t:
                summary, avg_logprobs, wers_machine, wers_machine_kw, wers_human_kw = get_data(c.output_path, c.dataset_type)
            print(f"Reading the generated files took: {t():.4f} s")

            for s, metric in zip(summary, [avg_logprobs, wers_machine, wers_machine_kw, wers_human_kw]):
                plot_metrics([metric],
                            f"Average {s["metric_name"]}s",
                             s["metric_name"],
                             [getattr(c, list_field)],
                             c.output_path)
            plot_correlations(wers_human_kw, wers_machine_kw, avg_logprobs, c)

            summary_arr.append(summary)
            avg_logprobs_arr.append(avg_logprobs)
            wers_machine_arr.append(wers_machine)
            wers_machine_kw_arr.append(wers_machine_kw)
            wers_human_kw_arr.append(wers_human_kw)

        for s_arr, metric_arr in zip(zip(*summary_arr), [avg_logprobs_arr, wers_machine_arr, wers_machine_kw_arr, wers_human_kw_arr]):
            plot_metrics(metric_arr,
                        f"Average {s_arr[0]['metric_name']}s",
                         s_arr[0]["metric_name"],
                         [c.model_type for c in unfolded_configs],
                         config.output_path)
        logger.info("Finished evaluation")


if __name__ == '__main__':
    evaluate_run(Path("inferences/dummy_data3"))