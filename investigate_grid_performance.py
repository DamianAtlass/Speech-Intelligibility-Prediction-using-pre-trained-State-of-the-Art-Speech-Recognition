import torch
from pathlib import Path

from utils.new_config_dataclass import InferenceConfig, load_config
from utils.evaluate_utils import get_data
from utils.evaluate_utils import get_kw_using_mixed_approaches
from utils.evaluate_utils import join_kw_list
from utils.wer_needleman_wunsch import wer_needleman_wunsch
from utils.werpy_utils import normalize
path = Path("inferences/turbo_default_grid_subwords")
config: InferenceConfig = load_config(path/"config.yaml")


df = get_data(
    config.model.name,
    config.output_path,
    config.data.val_split.dataset_type,
    config.extract_logprobs,
    config.word_timestamps,torch.device("cpu"),)

import seaborn as sns
import jiwer
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. Full-transcript WER (recompute directly, don't trust only the
#    precomputed column — useful sanity check on normalization)
# ---------------------------------------------------------------

#df_clean = df[df["snr"] == 40].copy()
#df_clean = df_clean[df["wers_machine_kw"]==1]

n_keywords = 3
kw_label = ["color", "letter", "digit"]
fig, axes = plt.subplots(n_keywords, 1, figsize=(12, 4 * n_keywords))

for pos, label in enumerate(kw_label):
    sub_pairs = Counter()

    kw_specific_wer = wer_needleman_wunsch(
        references=[o.split()[pos] for o in df["references_kw"]],
        transcripts=[o[pos] for o in df["machine_trans_kw_from_time_align"]])
    kw_specific_wer = round(kw_specific_wer*100, 2)

    for _, row in df.iterrows():
        ref = row["references_kw"].split()[pos]
        hyp = normalize([row["machine_trans_kw_from_time_align"][pos]], apply_werpy_normalize=False, apply_separate_numbers_from_letter=False)[0]

        ref = str(ref).strip().lower() if ref is not None else None
        hyp = str(hyp).strip().lower() if hyp is not None else None

        if ref is None or ref == "":
            continue
        if ref != hyp:
            sub_pairs[(ref, hyp)] += 1

    top_subs = pd.Series(sub_pairs).sort_values(ascending=False).head(7)
    top_subs.index = [f"{r} → {h}" for r, h in top_subs.index]

    ax = axes[pos]
    top_subs.plot(kind="barh", ax=ax, color="tab:purple")
    ax.invert_yaxis()
    ax.set_title(f"Top mismatches — keyword: {label}, WER: {kw_specific_wer}%")
    ax.set_xlabel("Count")


print(f"wer: {wer_needleman_wunsch(
    references=df["references_kw"],
    transcripts=join_kw_list(df["machine_trans_kw_from_time_align"])
):}")
fig.tight_layout()
plt.savefig("file.png")
plt.show()
