from datasets import DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer,EarlyStoppingCallback, IntervalStrategy
import torch
from whisper import available_models
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
metric = evaluate.load("wer")
from utils.config_dataclasses import TrainingConfig
from utils.logging_utils import capture_stdout, catch_time
import os
import json
from math import ceil
from utils.dataset_utils import get_dataset, apply_split

import logging
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def train_whisper(config: TrainingConfig, dataset: DatasetDict, device: torch.device):

    if "val" in dataset.keys():
        dataset.pop("val")
    del dataset
    train_dataset = get_dataset("grid", add_noise=True)
    train_dataset = apply_split(train_dataset, train_split=1., test_split=0, val_split=0)

    test_dataset = get_dataset("grid_bc", add_noise=False)
    test_dataset = apply_split(test_dataset, train_split=0, test_split=.2, val_split=0)

    dataset = DatasetDict(
        {"train": train_dataset["train"],
         "test": test_dataset["test"], }
    )

    full_model_name = f"{config.model}-{config.model_type}"

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN!")

    if config.model != "whisper":
        raise ValueError("Wrong model.")

    if config.model_type not in available_models():
        raise ValueError("That is not an available model!")
    logger.info("load feature extractor")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{full_model_name}", token=hf_token)

    language = None if "en" in config.model_type else "English"
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/{full_model_name}",
                                                 language=language,
                                                 task="transcribe",
                                                 token=hf_token)

    processor = WhisperProcessor.from_pretrained(f"openai/{full_model_name}",
                                                 language=language,
                                                 task="transcribe",
                                                 token=hf_token)

    #dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) pretty sure thats not needed, see parsing
    logger.info("prepare dataset")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = \
        feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    with catch_time() as t:
        dataset = dataset.map(
            prepare_dataset,
            num_proc=4,
            load_from_cache_file=False,
        )# set cache to false for debugging

        for split in dataset:
            cols_to_remove = [
                c for c in dataset[split].column_names
                if c not in {"input_features", "labels"}
            ]
            dataset[split] = dataset[split].remove_columns(cols_to_remove)

    dataset.set_transform(lambda x: x) # reset transformation

    logger.info(f"Execution time of dataset mapping: {t()/60:.4f} min")

    logger.info(f"Define model")

    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/{full_model_name}",
        token=hf_token,
        torch_dtype=torch.float32
    )
    model.to(device)
    if language is not None:
        model.generation_config.language = language
        model.generation_config.task = "transcribe"

    # if you ever decide to calculate timestamps with the hf model, remember to set accordingly model.generation_config.alignment_heads
    # https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a
    # see last line of whisper/__init__.load_model()

    model.generation_config.forced_decoder_ids = None

    logger.info(f"Define data collector")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    num_training_batches = ceil(len(dataset["train"])/config.batch_size)
    logger.info(f"Training batches per epoch: {num_training_batches}")
    logger.info(f"Test batches: {ceil(len(dataset["test"])/config.batch_size)}")

    foo = ceil(num_training_batches/config.save_and_eval_steps)
    logger.info(f"Saves/evaluations per epoch: {config.save_and_eval_steps}")


    logger.info(f"Define training args")
    # step equals batch, more or less
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_path),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        gradient_checkpointing=True, # reduces speed but allows for bigger models
        fp16=True,
        eval_strategy="steps",
        eval_steps=foo,
        save_strategy="steps",
        save_steps=foo,
        per_device_eval_batch_size=config.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=15,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        num_train_epochs=config.num_train_epochs,
        eval_on_start=False,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    logger.info(f"Define trainer")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    if config.perform_training:

        logger.info(f"Start training")
        with capture_stdout(logger.info, __name__):
            train_output = trainer.train()
        logger.info(f"Training done")

        summary = train_output._asdict()
        with open(config.output_path / "summary.json", 'w') as f:
            json.dump({"summary:": summary}, f, indent=4)
    else:
        logger.info(f"Skip training")

    trainer.save_model(str(config.output_path))

if __name__ == '__main__':
        pass
