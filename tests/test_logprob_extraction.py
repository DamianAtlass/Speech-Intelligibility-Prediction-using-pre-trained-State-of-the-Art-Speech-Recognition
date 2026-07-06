from inference import inference
import pytest
from pathlib import Path

from utils.config_dataclasses import InferenceConfig
from utils.grid_utils import get_grid
from utils.dataset_utils import apply_split
from dotenv import load_dotenv
load_dotenv() # needs to be before 'import torch'!
import torch
import shutil
import json
from whisper.tokenizer import get_tokenizer
from utils.werpy_utils import normalize
from utils.evaluate_utils import get_only_keywords_using_alignments


test_folder = Path.cwd() / "tests" if Path.cwd().name != "tests" else Path.cwd()


def test_logprob_extraction():

    config = InferenceConfig(
        model="whisper",
        model_type="tiny",
        model_path=None,
        output_path=test_folder/"logprob_extraction",
        dataset_type="grid",
        dataset_path=test_folder.parent/"datasets/grid/",
        train_split=0,
        test_split=0,
        val_split=1,
        extract_logprobs=True,
        word_timestamps=False,
        beam_size=5
    )
    if config.output_path.exists():
        shutil.rmtree(config.output_path)

    dataset = get_grid(config.dataset_path)
    dataset = apply_split(dataset, config.train_split, config.test_split, config.val_split, config.dataset_scaling)
    config.output_path.mkdir(exist_ok=config.debug)
    inference(config, dataset, torch.device("cuda"))
    with open(str(test_folder/"logprob_extraction"/"data"/"s7_bwas7s.json")) as f:
        data = json.load(f)
    tensor = torch.load(test_folder/"logprob_extraction"/"logprobs"/"s7_bwas7s.pt")
    print()

    tokens = data["prediction_result"]["segments"][0]["tokens"]
    tokenizer = get_tokenizer(multilingual=True)
    decoded_tokens = [tokenizer.decode_with_timestamps([t]) for t in tokens]

    decoded_tokens = decoded_tokens[1:-1]
    tensor = tensor[1:-1]# cut timestamp tokens primitively here


    decoded_tokens = normalize(decoded_tokens,
        apply_separate_numbers_from_letter=False,
        apply_werpy_normalize=False)
    decoded_tokens = [o.lower().strip() for o in decoded_tokens]

    trans_keywords_indices = get_only_keywords_using_alignments(
        reference=data["sentence"].split(),
        transcript=decoded_tokens,
        return_idx=True)

    #ref_align =   [None,   'bin', 'white', 'at', 's', 'seven', 'soon', None]
    #trans_align = ['been', 'why',    'it', "'s", 's', 'seven', 'soon',  '.']
    # -> keywords are at index 2, 4 and 5 in the transcript
    assert trans_keywords_indices == [2, 4, 5]