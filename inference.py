from tqdm import tqdm
import torch
import sip_whisper
from datasets import Dataset, DatasetDict

from utils.evaluate_utils import get_kw_idx_through_time_alignments, ref_alignments_to_seconds_and_rm_non_words
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

from utils.variables import *
from utils.werpy_utils import normalize

from utils.parakeet_utils import get_collate_fn
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
logger = logging.getLogger(__name__)

def create_filename(dataset_type: str, sample: dict, run: int|None, dispersion: bool, forced_alignment_options: dict|None) -> str:
    audio_path = Path(sample["audio_path"]) if "audio_path" in sample.keys() else None
    l = []

    if dispersion:
        if forced_alignment_options is None:
            l.append("forced_alignment-None")
        else:
            word: str = forced_alignment_options["token_id_or_word"].strip()
            for idx, values in zip([1,3,4], grid_kw_vocab.values()):
                if word not in values:
                    continue
                kw_idx = idx
                word_idx = values.index(word)

            assert isinstance(kw_idx, int)
            assert isinstance(word_idx, int)
            l.append(f"forced_alignment-{kw_idx}-{word_idx}")

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
                ) -> dict:
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

    logger.info(f"Save result data for sample [{counter}] to {result_data_file_path.relative_to(Path.cwd())}")

    with open(result_data_file_path, 'w') as f:
        json.dump(sample_dict, f, indent=4, cls=NpEncoder)

    if seperator: logger.info("--------------------")
    return sample_dict

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
                        if config.dispersion:
                            inference_whisper_with_forced_alignment(model, config, sample, device, run, counter)
                        else:
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

def inference_whisper(model,
                      config: InferenceConfig,
                      sample: dict,
                      device: torch.device,
                      run: int,
                      counter: int,
                      forced_alignment_options: dict|None = None
                      ) -> dict:

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
        "forced_alignment_options": None if forced_alignment_options is None else forced_alignment_options.copy()
    }

    # results will not be equal (therefore not deterministic) if temperature=!0.0, bc of stochastic sampling
    #  https://github.com/openai/whisper/discussions/81
    result: dict = sip_whisper.transcribe(**options)

    file_name = create_filename(
        dataset_type=config.data.val_split.dataset_type,
        sample=sample,
        run=run if config.runs_per_sample > 1 else None,
        dispersion=config.dispersion,
        forced_alignment_options=forced_alignment_options)

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

    if config.dispersion:
        sample.update({"varrying_transcription_options": {"forced_alignment_options": forced_alignment_options}})

    if config.runs_per_sample > 1:
        sample.update({"varrying_transcription_options": {"run": run}})

    result_data_file_path = config.output_path / "data" / f"{file_name}.json"
    result_data = save_result(sample.copy(), result, result_data_file_path, counter, True)

    return result_data

def inference_whisper_with_forced_alignment(
        model,
        config: InferenceConfig,
        sample: dict,
        device: torch.device,
        run: int,
        counter: int,):

    # regular run:
    regular_run_data = inference_whisper(model, config, sample, device, run, counter)
    #############################################
    #############################################
    keywords, kw_token_idx = get_kw_dirty(regular_run_data)

    keywords_norm = normalize(keywords)

    l = len(grid_all_keywords) - sum([1 for i,v in enumerate(grid_kw_vocab.values()) if keywords_norm[i] in v])
    counter = 1

    for i, kw_label in enumerate(grid_kw_labels):
        kw_pos = kw_token_idx[i]
        kw_for_forced_alignment = grid_kw_vocab[kw_label].copy()
        if keywords_norm[i] in kw_for_forced_alignment:
            kw_for_forced_alignment.remove(keywords_norm[i])

        for kw in kw_for_forced_alignment:
            forced_alignment_options = {"position": kw_pos, "token_id_or_word": " " + kw, }
            logger.info(f"{forced_alignment_options = } ({counter}/{l})")
            inference_whisper(model, config, sample, device, run, counter, forced_alignment_options)
            counter+=1


def get_kw_dirty(data: dict) -> tuple[list[str], list[list[int] | None]]:
    """
    Uses the same workflow as in evaluate_utils.py to get the right kw positions / tokens.

    """
    decoded_tokens_with_timestamps = data["prediction_result"]["decoded_tokens_with_timestamps"]
    no_timestamp_idx = ["<|" not in t and "|>" not in t for t in decoded_tokens_with_timestamps]
    decoded_tokens_without_timestamp_tokens = [t for t, b in zip(decoded_tokens_with_timestamps, no_timestamp_idx) if b]
    del decoded_tokens_with_timestamps, no_timestamp_idx

    ref_alignments = [{"start": a[0], "end": a[1], "word": a[2]} for a in data["alignment"]]
    ref_alignments: list[dict] = ref_alignments_to_seconds_and_rm_non_words(ref_alignments)

    transcript_alignment = []
    for s in data["prediction_result"]["segments"]:
        transcript_alignment.extend(s["words"])
    for o in transcript_alignment:
        o["word"] = normalize([o["word"]], apply_separate_numbers_from_letter=False, apply_werpy_normalize=False, )[0]

    kw_token_idx_from_alignment, _ = get_kw_idx_through_time_alignments(
        reference_alignments=ref_alignments,
        transcript_alignments=transcript_alignment)

    kw_token_idx = [o[0] for o in kw_token_idx_from_alignment]
    keywords = [decoded_tokens_without_timestamp_tokens[o] for o in kw_token_idx]
    return keywords, kw_token_idx

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