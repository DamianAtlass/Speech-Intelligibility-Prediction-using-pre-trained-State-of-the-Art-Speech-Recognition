from pathlib import Path
import json
import csv
from utils.paths import CPC_DEMO_FOLDER, PROJECT_ROOT
from datasets import Dataset, load_from_disk, Audio
from utils.manipulate_audio import calculate_snr


WANTED_SAMPLE_RATE = 16_000

def parse_and_save_cpc(cpc_folder: Path = CPC_DEMO_FOLDER):
    metadata_path = cpc_folder / "metadata"

    with open(metadata_path/"CPC3.train.json", "r") as file:
        listening_test_results = json.load(file)

    with open(metadata_path/"listeners.csv") as metadata_file:
        data = csv.reader(metadata_file)
        listeners_to_HI_severity = {}
        for row in data:
            k,v = row
            if k=="severity" and v=="severity":
                continue
            listeners_to_HI_severity[k] = v

    for r in listening_test_results:
        challenge_number, enhancement_model, scene, listener = r["signal"].split("_")

        r["listener"] = listener
        audio_path: Path = (CPC_DEMO_FOLDER / "train"/ "signals" / f"{r["signal"]}.wav").relative_to(PROJECT_ROOT)
        ref_audio_path: Path = (CPC_DEMO_FOLDER / "train"/ "references" / f"{challenge_number}_{scene}_ref.wav").relative_to(PROJECT_ROOT)

        r["reference_audio_path"] = str(ref_audio_path)
        r["audio_path"] = str(audio_path)

        #will be converted into lazy row later
        r["reference_audio"] = str(ref_audio_path)
        r["audio"] = str(audio_path)

        assert audio_path.exists()
        assert ref_audio_path.exists()
        r["hearing_impairment_severity"] = listeners_to_HI_severity[r["listener"]]

    del listeners_to_HI_severity

    d = {k: [o[k] for o in listening_test_results] for k in listening_test_results[0].keys()}
    dataset = (Dataset.from_dict(d)
               .cast_column("reference_audio", Audio(sampling_rate=WANTED_SAMPLE_RATE))
               .cast_column("audio", Audio(sampling_rate=WANTED_SAMPLE_RATE)))

    snr_array: list[float] = []
    for sample in dataset:

        audio = sample["audio"]["array"]
        clean_audio = sample["reference_audio"]["array"]

        # audio and clear_audio sometimes have a length difference of 1
        min_length = min(len(audio), len(clean_audio))
        audio = audio[:min_length]
        clean_audio = clean_audio[:min_length]

        noise = audio - clean_audio

        snr = calculate_snr(signal=audio, noise=noise)
        snr_array.append(float(snr))

    dataset = dataset.add_column("snr", snr_array)

    save_at = cpc_folder / "saved_dataset"
    dataset.save_to_disk(save_at)


if __name__ == '__main__':
    pass