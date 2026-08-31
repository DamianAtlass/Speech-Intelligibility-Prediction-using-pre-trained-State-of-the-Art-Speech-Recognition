from tqdm import tqdm
import torch
import sip_whisper
from datasets import Dataset, DatasetDict
from utils.new_config_dataclass import InferenceConfig
from typing import Any, cast
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

def create_filename(dataset_type: str, sample: dict, run: int|None) -> str:
    audio_path = Path(sample["audio_path"]) if "audio_path" in sample.keys() else None
    l = []
    match dataset_type:
        case "grid_bc":
            snr_str = f"{"m" if int(sample["snr_db"]) < 0 else ""}{abs(int(sample["snr_db"]))}"
            l.append(f"SNR{snr_str}")
            l.append(f"l{sample["listener"]}")
            l.append(f"s{sample["speaker"]}")
            tail = f"{audio_path.stem.split("_")[1]}"
        case "grid":
            l.append(f"s{sample["speaker"]}")
            tail = audio_path.stem
        case "libri":
            tail = sample["id"]
        case _:
            raise RuntimeError("Unknown dataset type")
    if run is not None:
        l.append(f"run{run}")

    l.append(tail)
    filename = "_".join(l)
    return filename

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.item()
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
    if result_data_file_path.exists(): raise FileExistsError
    sample_dict.pop("audio")
    sample_dict["prediction_result"] = result

    logger.info(f"Save result data [{counter}] to {result_data_file_path.relative_to(Path.cwd())}")

    with open(result_data_file_path, 'w') as f:
        json.dump(sample_dict, f, indent=4, cls=NpEncoder)

    if seperator: logger.info("--------------------")

def inference_loop_whisper(config: InferenceConfig, model: Any, dataset: Dataset, device: torch.device) -> None:
    """
    Perform inference on a batch of data to collect the results and certain internal values. Saves it locally file by file.
    config: InferenceConfig
    model: something like torch.nn.Module
    dataset: Dataset
    device: torch.device

    Returns: None
    """
    if config.temperature!=0 and config.beam_size!=1:
        raise ValueError("Beamsize would be overridden for temperature !=1!")

    exception_log: list[int] = []
    with torch.inference_mode():
        num_total_transciptions = len(dataset) * config.runs_per_sample
        with tqdm(total=num_total_transciptions) as pbar:
            counter = 0
            for idx_sample, sample in enumerate(dataset):
                for run in range(config.runs_per_sample):
                    logger.info(f"Transcribe {sample["audio_path"] if "audio_path" in sample.keys() else sample["id"]} (#{idx_sample}) "
                                f"{f"(run: {run})" if config.runs_per_sample > 1 else ""} ...")
                    try:
                        inference_whisper(model, config, sample, device, run, counter)
                    except Exception as e:
                        logger.critical(e)
                        exception_log.append(counter)
                    pbar.update(1)
                    counter+=1

    if len(exception_log) == 0:
        logger.info("No exceptions occured.")
    else:
        logger.critical(f"{len(exception_log)} exceptions occured! ({round(len(exception_log)/num_total_transciptions, 3)*100}%)")
        logger.critical(f"Bad transcriptions at counter= {exception_log}")

def inference_whisper(
        model,
        config: InferenceConfig,
        sample: dict,
        device: torch.device,
        run: int,
        counter: int,):

    audio_array = torch.tensor(sample["audio"]["array"]).to(device)
    audio_array = sip_whisper.pad_or_trim(audio_array)

    options = {
        "model": model,
        "audio": audio_array,
        "fp16": False,
        "beam_size": config.beam_size,
        "temperature": config.temperature,
        "extract_logprobs": config.extract_logprobs,
        "word_timestamps": config.word_timestamps,
        "condition_on_previous_text": False,
        "language": "en",
        "break_after_first_segment": False if config.model.path is None else True,  # only with trained models
        "subword_timestamps": config.subword_timestamps,
    }

    # results will not be equal (therefore not deterministic) if temperature=!0.0, bc of stochastic sampling
    #  https://github.com/openai/whisper/discussions/81
    result: dict = sip_whisper.transcribe(**options)

    file_name = create_filename(
        dataset_type=config.data.val_split.dataset_type,
        sample=sample, run=run if config.runs_per_sample > 1 else None)

    if config.extract_logprobs:
        extracted_logprobs = result.pop("extracted_logprobs")
        tokens = [x for s in result["segments"] for x in s["tokens"]]
        assert len(tokens) == (
            0 if extracted_logprobs is None else extracted_logprobs.shape[0]), "Missmatch of logprob and token length"

        if extracted_logprobs is not None:
            logprob_file_path = config.output_path / "logprobs" / f"{file_name}.pt"
            if logprob_file_path.exists(): raise FileExistsError
            logger.info(f"Save logprobs to {logprob_file_path}")
            torch.save(extracted_logprobs, logprob_file_path)
            result["logprobs_path"] = str(logprob_file_path.relative_to(Path.cwd()))

        else:
            result["logprobs_path"] = ""
            logger.info("No logprobs to save.")

    if config.runs_per_sample > 1:
        sample.update({"varrying_transcription_options": {"run": run}})

    result_data_file_path = config.output_path / "data" / f"{file_name}.json"
    save_result(sample.copy(), result, result_data_file_path, counter, True)

def inference_parekeet(config: InferenceConfig, model: EncDecCTCModelBPE, dataset: Dataset, device: torch.device) -> None:
    with torch.inference_mode():
        batch_size = 20
        counter = 0

        alignment_path = config.output_path/"alignments"
        alignment_path.mkdir()

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
                    timestamps=config.word_timestamps,
                    verbose=False
                )
                for sample, result in zip(subset, transcriptions):
                    result = asdict(result)

                    print("TODO needs for loop for runs")
                    result_file_name = create_filename(config.data.val_split.dataset_type, sample, None)

                    result_data_file_path = config.output_path / "data" / f"{result_file_name}.json"

                    alignments_data_file_path = alignment_path / f"{result_file_name}.pt"
                    if alignments_data_file_path.exists(): raise FileExistsError
                    torch.save(result.pop("alignments"), alignments_data_file_path)

                    result["alignments_path"] = str(alignments_data_file_path.relative_to(Path.cwd()))

                    result["y_sequence"] = [int(a) for a in result["y_sequence"]]
                    result["score"] = result["score"]#.item()

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
    if config.model.name =="whisper":
        inference_loop_whisper(config, model, dataset["val"], device)
    elif config.model.name =="parakeet":
        inference_parekeet(config, model, dataset["val"], device)
    else:
        raise NotImplementedError
    print()

if __name__ == '__main__':
    pass