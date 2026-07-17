from utils.config_dataclasses import get_config, unfold_config, InferenceConfig
from pathlib import Path
from utils.logging_utils import catch_time

from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch' to control what gpu to use (since some libs chose automatically)!
import torch
import logging
logger = logging.getLogger(__name__)
from utils.evaluate_utils import get_data, evaluate_individual_run
from utils.plotting_utils import plot_metrics, boxplot_corr_per_listener, plot_wer_to_snr
import pandas as pd
from utils.cuda_utils import select_device


def evaluate_run(path: Path, device: torch.device | None = None):
    if not device:
        device = select_device()

    config = get_config(path / "config.ini")
    unfolded_configs: list[InferenceConfig] = unfold_config(config)

    if len(unfolded_configs)==1:
        with catch_time() as t:
            summary, df_single_run = get_data(config.output_path, config.dataset_type, config.extract_logprobs, device)
        print(f"Reading the generated files took: {t():.4f} s")
        df_single_run["model_type"] = config.model_type
        evaluate_individual_run(config, summary, df_single_run, device)

        #with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", 10000):
            #print(df_single_run[["references_kw", "machine_transcripts", "machine_transcripts_kw", "wers_machine_kw"]])
    else:
        summary_arr, avg_logprobs_arr, wers_machine_arr, wers_machine_kw_arr = [], [], [], []
        #list_field = config.get_list_fields()[0]

        df_all = None

        for c in unfolded_configs:
            with catch_time() as t:
                summary, df_single_run = get_data(c.output_path, c.dataset_type, c.extract_logprobs, device=device)
                df_single_run["model_type"] = c.model_type

                if df_all is None:
                    df_all = df_single_run
                else:
                    df_all = pd.concat([df_all, df_single_run], ignore_index=True)
            print(f"Reading the generated files took: {t():.4f} s")

            evaluate_individual_run(c, summary, df_single_run, device)

            summary_arr.append(summary)
            avg_logprobs_arr.append(df_single_run["avg_logprobs"])
            wers_machine_arr.append(df_single_run["wers_machine"])
            wers_machine_kw_arr.append(df_single_run["wers_machine_kw"])



        # plot group plots
        for s_arr, metric_arr in zip(zip(*summary_arr[:-1]), [avg_logprobs_arr, wers_machine_arr, wers_machine_kw_arr]):
            plot_metrics(metric_arr,
                        f"Average {s_arr[0]['metric_name']}",
                         s_arr[0]["metric_name"],
                         [c.model_type for c in unfolded_configs],
                         config.output_path)


        if config.dataset_type != "grid":
            # plot_wer_to_snr(df_all[["wers_human_kw", "wers_machine_kw", "snr", "model_type"]],
            #                 config,
            #                 shifting_attribute="model_type",
            #                 output_path=config.output_path)


            boxplot_corr_per_listener(df_all[["wers_human_kw", "wers_machine_kw", "model_type", "listener"]],
                                      correlate_to="wers_machine_kw",
                                      model = config.model,
                                      model_type = config.model_type,
                                      output_path = config.output_path)

            boxplot_corr_per_listener(df_all[["wers_human_kw", "avg_logprobs", "model_type", "listener"]],
                                      correlate_to="avg_logprobs",
                                      model=config.model,
                                      model_type=config.model_type,
                                      output_path=config.output_path)

    logger.info("Finished evaluation")


if __name__ == '__main__':
    evaluate_run(Path("inferences/small.en60"))