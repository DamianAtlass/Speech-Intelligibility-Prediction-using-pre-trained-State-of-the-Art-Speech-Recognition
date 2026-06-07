from utils.config_dataclasses import InferenceConfig, get_config, unfold_config
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
import matplotlib.pyplot as plt


def plot_metric(array: list[torch.Tensor],
                title: str,
                metric_name: str,
                x_label: list[str],
                output_path: Path | None
                ) -> None:
    #sort

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

def mtd(t: torch.Tensor):
    t = torch.exp(t)
    t1 = t[:,:-1]
    t2 = t[:,1:]

    t3 = t1 - t2

    t3 = torch.sum(t3**2, dim=1)
    return t3.sum()


def get_data(output_path: Path) -> tuple:
    data_path = output_path / "data"
    device = select_device()

    avg_logprobs = []
    references = []
    transcripts = []

    for file in data_path.iterdir():
        with open(file) as f:
            json_file = json.load(f)

            avg_logprobs.append(json_file["result"]["segments"][0]["avg_logprob"])
            references.append(json_file["sentence"])
            transcripts.append(json_file["result"]["text"])

    transcripts = additional_normalization(transcripts)

    wers = calculate_wers_with_norm(transcripts, references).to(device)
    avg_logprobs = torch.Tensor(avg_logprobs).to(device)

    summary = []
    for name, values in zip(["Logprob(per sequence)", "WER"], [avg_logprobs, wers]):
        summary.append({
            "metric_name": name,
            "mean": values.mean().item(),
            "median": values.median().item(),
            "std": values.std().item(),
            "n": values.shape[0]
        })

    with open(output_path/"summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    return summary, avg_logprobs, wers

def evaluate_run(path: Path):

    config = get_config(path / "config.ini")

    unfolded_configs = unfold_config(config)

    if len(unfolded_configs)==1:
        with catch_time() as t:
            summary, avg_logprobs, wers = get_data(config.output_path)
        print(f"Reading the generated files took: {t():.4f} s")

        for s, metric in zip(summary, [avg_logprobs, wers]):
            plot_metric(metric, f"Average {s["metric_name"]}s for {config.model}({config.model_type})", s["metric_name"], [c.model_type for c in unfolded_configs], config.output_path)

    else:
        summary_arr, avg_logprobs_arr, wers_arr = [], [], []
        list_field = config.get_list_fields()[0]

        for c in unfolded_configs:
            with catch_time() as t:
                summary, avg_logprobs, wers = get_data(c.output_path)
            print(f"Reading the generated files took: {t():.4f} s")

            for s, metric in zip(summary, [avg_logprobs, wers]):
                plot_metric([metric],
                            f"Average {s["metric_name"]}s",
                            s["metric_name"],
                            [getattr(c, list_field)],
                            c.output_path)

            summary_arr.append(summary)
            avg_logprobs_arr.append(avg_logprobs)
            wers_arr.append(wers)

        for s_arr, metric_arr in zip(zip(*summary_arr), [avg_logprobs_arr, wers_arr]):
            plot_metric(metric_arr,
                        f"Average {s_arr[0]['metric_name']}s",
                        s_arr[0]["metric_name"],
                        [c.model_type for c in unfolded_configs],
                        config.output_path)


if __name__ == '__main__':
    evaluate_run(Path("inferences/full_data_all_models"))