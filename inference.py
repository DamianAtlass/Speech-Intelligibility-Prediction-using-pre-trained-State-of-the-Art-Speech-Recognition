import tqdm
import torch
import whisper
import sip_whisper
from utils.convert_hf_to_openai_format import load_whisper_model
from datasets import Dataset, DatasetDict
from utils.config_dataclasses import InferenceConfig
from typing import Any
import json
from pathlib import Path

def batch_inference(config: InferenceConfig, model: Any, dataset: Dataset) -> None:

    if config.extract_logits:
        (config.output_path/"logits").mkdir(exist_ok=config.debug)
    (config.output_path / "data").mkdir(exist_ok=config.debug)

    for sample in tqdm.tqdm(dataset):

        sample_dict = (dict(sample))
        sample_dict.pop("audio")
        audio_array = torch.tensor(sample["audio"]["array"])

        # results will not be equal (therefore not deterministic) if temperature=!0.0
        #  https://github.com/openai/whisper/discussions/81
        if config.extract_logits:
            result: dict = sip_whisper.transcribe(model, audio_array, fp16=False, beam_size=config.beam_size, temperature=0, word_timestamps=config.word_timestamps, condition_on_previous_text=False)
        else:
            result: dict = whisper.transcribe(model, audio_array, fp16=False, beam_size=config.beam_size, temperature=0, word_timestamps=config.word_timestamps, condition_on_previous_text=False)

        audio_path = Path(sample["audio_path"])
        file_name = f"{audio_path.parent.stem}_{audio_path.stem}"

        if config.extract_logits:
            extracted_logprobs = result.pop("extracted_logprobs")
            file_path = config.output_path/"logits"/f"{file_name}.pt"
            torch.save(extracted_logprobs, file_path)

            result["logprobs_path"] = str(file_path)

        sample_dict["result"] = result
        file_path = config.output_path / "data" / f"{file_name}.json"
        with open(file_path, 'w') as f:
            json.dump(sample_dict, f, indent=4)

        break

def inference(config: InferenceConfig, dataset: DatasetDict):
    model = load_whisper_model(config)

    batch_inference(config, model, dataset["val"])
    print()

if __name__ == '__main__':
    pass