import tqdm
import torch
import whisper
import sip_whisper
from datasets import Dataset, DatasetDict
from utils.config_dataclasses import InferenceConfig
from typing import Any
import json
from pathlib import Path
import logging
from utils.models_utils import load_model
from torch.utils.data import DataLoader
from math import ceil

from utils.parakeet_utils import collate
from utils.logging_utils import catch_time
logger = logging.getLogger(__name__)

def inference_whisper(config: InferenceConfig, model: Any, dataset: Dataset, device: torch.device) -> None:
    """
    Perform inference on a batch of data to collect the results and certain internal values. Saves it locally file by file.
    config: InferenceConfig
    model: something like torch.nn.Module
    dataset: Dataset
    device: torch.device

    Returns: None
    """
    transcribe_fn = sip_whisper.transcribe if config.extract_logprobs else whisper.transcribe
    counter = 0

    with torch.inference_mode():
        for sample in tqdm.tqdm(dataset):
            counter = counter + 1
            logger.info(f"Transcribe {sample["audio_path"]}...")

            sample_dict = dict(sample)
            sample_dict.pop("audio")
            audio_array = torch.tensor(sample["audio"]["array"]).to(device)
            audio_array = whisper.pad_or_trim(audio_array)

            options = {
                "model": model,
                "audio": audio_array,
                "fp16": False,
                "beam_size": config.beam_size,
                "temperature": 0,
                "word_timestamps": config.word_timestamps,
                "condition_on_previous_text": False,
                "language": "en"
            }

            # results will not be equal (therefore not deterministic) if temperature=!0.0
            #  https://github.com/openai/whisper/discussions/81
            result: dict = transcribe_fn(**options)

            audio_path = Path(sample["audio_path"])
            file_name = f"{audio_path.parent.stem}_{audio_path.stem}"

            if config.extract_logprobs:
                extracted_logprobs = result.pop("extracted_logprobs")
                tokens = [x for s in result["segments"] for x in s["tokens"]]
                assert len(tokens) == (0 if extracted_logprobs is None else extracted_logprobs.shape[0]), "Missmatch of logprob and token length"

                if extracted_logprobs is not None:
                    logprob_file_path = config.output_path/"logprobs"/f"{file_name}.pt"
                    logger.info(f"Save logprobs to {logprob_file_path}")
                    torch.save(extracted_logprobs, logprob_file_path)
                    result["logprobs_path"] = str(logprob_file_path.relative_to(Path.cwd()))

                else:
                    result["logprobs_path"] = ""
                    logger.info("No logprobs to save.")

            sample_dict["prediction_result"] = result
            result_data_file_path = config.output_path / "data" / f"{file_name}.json"
            logger.info(f"Save result data [{counter}] to {result_data_file_path.relative_to(Path.cwd())}")
            with open(result_data_file_path, 'w') as f:
                json.dump(sample_dict, f, indent=4)

            logger.info("--------------------")

def inference_parekeet(config: InferenceConfig, model: Any, dataset: Dataset, device: torch.device) -> None:
    with torch.inference_mode():
        batch_size = 20

        for i in range(ceil(len(dataset) / batch_size)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(dataset))
            subset = dataset.select(range(start, end))

            dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate)
            with catch_time() as t:
                transcriptions = model.transcribe(
                    audio=dataloader,
                    timestamps=True,
                )
            print(f"Execution time of do_something: {t():.1f} s")

            for sample, result in zip(subset, transcriptions):
                pass



def inference(config: InferenceConfig, dataset: DatasetDict, device: torch.device) -> None:
    if config.extract_logprobs:
        (config.output_path/"logprobs").mkdir(exist_ok=config.debug)
    (config.output_path / "data").mkdir(exist_ok=config.debug)

    model = load_model(config, device)

    #dataset["val"] = dataset["val"].filter(lambda sample: sample["audio_path"]=='datasets/GridIntelligibilityDatabase/BC2007wavs/BC2007/m12/6/s5_bbar7s.wav')
    if config.model =="whisper":
        inference_whisper(config, model, dataset["val"], device)
    elif config.model =="parakeet":
        inference_parekeet(config, model, dataset["val"], device)
    else:
        raise NotImplementedError
    print()

if __name__ == '__main__':
    pass