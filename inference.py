from tqdm import tqdm
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
from dataclasses import asdict
import numpy as np

from utils.parakeet_utils import get_collate_fn
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_result(sample_dict: dict,
                result: dict,
                result_data_file_path: Path,
                counter: int,
                seperator: bool = True
                ) -> None:
    """
    Args:
        sample_dict: dict of the sample (i.e. item of a dataset (or subset))
        result: Resulting dict of the transcription. Might need to be transformed before.
        result_data_file_path:
        counter:
        seperator: print/logg a seperating line

    Returns:
    """

    sample_dict.pop("audio")
    sample_dict["prediction_result"] = result

    logger.info(f"Save result data [{counter}] to {result_data_file_path.relative_to(Path.cwd())}")

    with open(result_data_file_path, 'w') as f:
        json.dump(sample_dict, f, indent=4, cls=NpEncoder)

    if seperator: logger.info("--------------------")

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
        for sample in tqdm(dataset):
            counter = counter + 1
            logger.info(f"Transcribe {sample["audio_path"]}...")

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
                    if logprob_file_path.exists(): raise FileExistsError
                    logger.info(f"Save logprobs to {logprob_file_path}")
                    torch.save(extracted_logprobs, logprob_file_path)
                    result["logprobs_path"] = str(logprob_file_path.relative_to(Path.cwd()))

                else:
                    result["logprobs_path"] = ""
                    logger.info("No logprobs to save.")

            sample_dict = dict(sample)
            sample_dict.pop("audio")
            sample_dict["prediction_result"] = result
            result_data_file_path = config.output_path / "data" / f"{file_name}.json"
            if result_data_file_path.exists(): raise FileExistsError
            logger.info(f"Save result data [{counter}] to {result_data_file_path.relative_to(Path.cwd())}")
            with open(result_data_file_path, 'w') as f:
                json.dump(sample_dict, f, indent=4)

            logger.info("--------------------")

def inference_parekeet(config: InferenceConfig, model: EncDecCTCModelBPE, dataset: Dataset, device: torch.device) -> None:
    with torch.inference_mode():
        batch_size = 20
        counter = 0

        alignment_path = config.output_path/"alignments"
        alignment_path.mkdir()
        y_sequence_path = config.output_path/"y_sequence"
        y_sequence_path.mkdir()

        print("enter parakeet inference")
        with tqdm(total=len(dataset)) as pbar:
            prev = 0
            for i in range(ceil(len(dataset) / batch_size)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(dataset))
                subset = dataset.select(range(start_idx, end_idx))

                dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=get_collate_fn(device))
                print("batch inference...")
                transcriptions = model.transcribe(
                    audio=dataloader,
                    timestamps=True,
                    verbose=False
                )
                for sample, result in zip(subset, transcriptions):
                    result = asdict(result)

                    audio_path = Path(sample["audio_path"])
                    result_file_name = f"{audio_path.parent.stem}_{audio_path.stem}"

                    result_data_file_path = config.output_path / "data" / f"{result_file_name}.json"
                    alignments_data_file_path = alignment_path / f"{result_file_name}.pt"
                    y_sequence_data_file_path = y_sequence_path / f"{result_file_name}.pt"
                    if result_data_file_path.exists(): raise FileExistsError
                    if alignments_data_file_path.exists(): raise FileExistsError
                    if y_sequence_data_file_path.exists(): raise FileExistsError

                    torch.save(result.pop("alignments"), alignments_data_file_path)
                    torch.save(result.pop("y_sequence"), y_sequence_data_file_path)
                    result["alignments_path"] = str(alignments_data_file_path.relative_to(Path.cwd()))
                    result["y_sequence_path"] = str(y_sequence_data_file_path.relative_to(Path.cwd()))

                    result["score"] = result["score"].item()
                    result["length"] = result["length"].item()

                    save_result(sample_dict=dict(sample),
                                result=result,
                                result_data_file_path=result_data_file_path,
                                counter=counter,
                                seperator=False)
                    counter+=1

                pbar.update(end_idx - prev)
                prev = end_idx

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