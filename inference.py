import tqdm
import torch
import whisper
import sip_whisper
from utils.convert_hf_to_openai_format import load_whisper_model
from utils.logging_utils import catch_time
from datasets import Dataset, DatasetDict
from utils.config_dataclasses import InferenceConfig
from typing import Any
import json
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def batch_inference(config: InferenceConfig, model: Any, dataset: Dataset, device: torch.device) -> None:
    """
    Perform inference on a batch of data to collect the results and certain internal values. Saves it locally file by file.
    config: InferenceConfig
    model: something like torch.nn.Module
    dataset: Dataset
    device: torch.device

    Returns: None
    """

    if config.extract_logprobs:
        (config.output_path/"logprobs").mkdir(exist_ok=config.debug)
    (config.output_path / "data").mkdir(exist_ok=config.debug)

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
            with catch_time() as t:
                # results will not be equal (therefore not deterministic) if temperature=!0.0
                #  https://github.com/openai/whisper/discussions/81
                result: dict = transcribe_fn(model, audio_array, fp16=False, beam_size=config.beam_size, temperature=0, word_timestamps=config.word_timestamps, condition_on_previous_text=False)
            logger.info(f"Transcription time for sample {counter}/{len(dataset)}: {t():.4f} secs")

            audio_path = Path(sample["audio_path"])
            file_name = f"{audio_path.parent.stem}_{audio_path.stem}"

            if config.extract_logprobs:
                extracted_logprobs = result.pop("extracted_logprobs")
                file_path = config.output_path/"logprobs"/f"{file_name}.pt"
                logger.info(f"Save logprobs to {file_path}")
                torch.save(extracted_logprobs, file_path)

                result["logprobs_path"] = str(file_path)

            sample_dict["result"] = result
            file_path = config.output_path / "data" / f"{file_name}.json"
            logger.info(f"Save result data to {file_path}")
            with open(file_path, 'w') as f:
                json.dump(sample_dict, f, indent=4)

            logger.info("--------------------")

def inference(config: InferenceConfig, dataset: DatasetDict, device: torch.device) -> None:
    model = load_whisper_model(config, device)

    batch_inference(config, model, dataset["val"], device)
    print()

if __name__ == '__main__':
    pass