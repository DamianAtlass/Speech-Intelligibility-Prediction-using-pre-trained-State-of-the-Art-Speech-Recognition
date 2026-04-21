import tqdm
import torch
import whisper
import sip_whisper
from utils.convert_hf_to_openai_format import load_whisper_model
from datasets import Dataset, DatasetDict
from utils.config_dataclasses import InferenceConfig
from typing import Any
import json

def batch_inference(config: InferenceConfig, model: Any, dataset: Dataset) -> None:
    results = []

    if config.extract_logits:
        (config.output_dir/"logits").mkdir(exist_ok=config.debug)

    for sample in tqdm.tqdm(dataset):
        sample_dict = (dict(sample))
        sample_dict.pop("audio")
        audio_array = torch.tensor(sample["audio"]["array"])

        # results will not be equal (therefore not deterministic) if temperature=!0.0
        #  https://github.com/openai/whisper/discussions/81
        if config.extract_logits:
            result: dict = sip_whisper.transcribe(model, audio_array, fp16=False, beam_size=5, temperature=0, word_timestamps=config.word_timestamps)
        else:
            result: dict = whisper.transcribe(model, audio_array, fp16=False, beam_size=5, temperature=0, word_timestamps=config.word_timestamps)

        if config.extract_logits:
            extracted_logprobs = result.pop("extracted_logprobs")
            file_name = config.output_dir/"logits"/"tmp_file_name.pt"
            torch.save(extracted_logprobs, file_name)

            result["logprobs_path"] = str(file_name)

        sample_dict["result"] = result
        results.append(sample_dict)

        if config.debug:
            break

    with open(config.output_dir/'data.json', 'w') as f:
        json.dump(results, f, indent=4)

    #with open(config.output_dir/'data.json') as f:
    #    d = json.load(f)

def inference(config: InferenceConfig, dataset: DatasetDict):
    model = load_whisper_model(config)

    batch_inference(config, model, dataset["val"])
    print()

if __name__ == '__main__':
    pass