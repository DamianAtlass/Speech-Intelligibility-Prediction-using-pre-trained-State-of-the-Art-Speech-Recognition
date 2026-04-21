import tqdm
import torch
import whisper
from utils.convert_hf_to_openai_format import load_whisper_model
from datasets import Dataset, DatasetDict
from utils.config_dataclasses import InferenceConfig
from typing import Any
import json


def batch_inference(config: InferenceConfig, model: Any, dataset: Dataset) -> None:
    results = []
    (config.output_dir/"logits").mkdir(exist_ok=config.debug)

    for sample in tqdm.tqdm(dataset):
        sample_dict = (dict(sample))
        sample_dict.pop("audio")
        audio_array = torch.tensor(sample["audio"]["array"])

        # results will not be equal (therefore not deterministic) if temperature=0.0
        #  https://github.com/openai/whisper/discussions/81
        result: dict = whisper.transcribe(model, audio_array, fp16=False, temperature=0.0, word_timestamps=True)
        sample_dict["result"] = result
        results.append(sample_dict)

        if config.debug:
            break

    with open(config.output_dir/'data.json', 'w') as f:
        json.dump(results, f)

    #with open(config.output_dir/'data.json') as f:
    #    d = json.load(f)

def inference(config: InferenceConfig, dataset: DatasetDict):
    model = load_whisper_model(config.model_path, explicit_model_type=config.model_type)

    batch_inference(config, model, dataset["val"])
    print()

if __name__ == '__main__':
    pass