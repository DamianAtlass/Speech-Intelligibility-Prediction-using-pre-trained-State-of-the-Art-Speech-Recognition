
from datasets import DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from whisper import available_models
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
metric = evaluate.load("wer")
from utils.config_dataclasses import TrainingConfig
#import datetime


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


def train_whisper(config: TrainingConfig, dataset: DatasetDict):
    #model_type = config.model
    full_model_name = f"{config.model}-{config.model_type}"
    #output_dir = f"{model_type}_{datetime.datetime.now().strftime("%d_%m_%Y-%H.%M.%S")}"

    if config.model != "whisper":
        raise ValueError("Wrong model.")

    if config.model_type not in available_models():
        raise ValueError("That is not an available model!")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{full_model_name}")

    lang_for_tokenizer = None if "en" in config.model_type else "English"
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/{full_model_name}", language=lang_for_tokenizer, task="transcribe")

    processor = WhisperProcessor.from_pretrained(f"openai/{full_model_name}", language="English", task="transcribe")

    #dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) pretty sure thats not needed, see parsing

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = \
        feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=12,
                          load_from_cache_file=True
                          )  # set cache to false for debugging

    model = WhisperForConditionalGeneration.from_pretrained(f"openai/{full_model_name}")
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"

    # if you ever decide to calculate timestamps with the hf model, remember to set accordingly model.generation_config.alignment_heads
    # https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a
    # see last line of whisper/__init__.load_model()

    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=config.learning_rate,
        warmup_steps=500,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="epoch",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="epoch",
        save_steps=1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        # max_steps=5,
        greater_is_better=False,
        num_train_epochs=config.num_train_epochs,
        eval_on_start=True

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


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    if config.perform_training:
        trainer.train()

    trainer.save_model(str(config.output_dir))

if __name__ == '__main__':
        train_whisper()
