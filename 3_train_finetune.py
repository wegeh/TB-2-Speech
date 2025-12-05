"""
Fine-tune pretrained Wav2Vec2 model for Javanese ASR.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import yaml
from datasets import Dataset
import soundfile as sf
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from jiwer import cer, wer
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    set_seed,
)

from src.dataset import create_splits, prepare_metadata


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for Javanese ASR.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_finetune.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("hf_cache"),
        help="Cache directory for Hugging Face downloads (helps avoid corrupted caches).",
    )
    return parser.parse_args()


def build_hf_datasets(metadata: Dict[str, Dataset], sample_rate: int):
    hf_splits = {}
    for split_name, df in metadata.items():
        if len(df) == 0:
            hf_splits[split_name] = None
            continue

        df_local = df.copy()
        df_local["filepath"] = df_local["filepath"].astype(str)

        # Keep paths only, do NOT cast to Audio()
        dataset = Dataset.from_pandas(df_local[["filepath", "transcript"]])
        dataset = dataset.rename_columns({"filepath": "audio_path"})

        # Remove any internal index columns (__index_level_0__ etc)
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col.startswith("__")]
        )

        hf_splits[split_name] = dataset
    return hf_splits


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    transcript_path = Path(data_cfg.get("transcript_path", "data/transcript_clean.csv"))
    audio_dir = Path(data_cfg.get("audio_dir", "data/wav_clean"))
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    seed = int(data_cfg.get("seed", 42))
    set_seed(seed)

    metadata = prepare_metadata(transcript_path, audio_dir)
    splits = create_splits(metadata, seed=seed)
    hf_splits = build_hf_datasets(splits, sample_rate=sample_rate)

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Change this in YAML to e.g. "cahya/wav2vec2-large-xlsr-javanese"
    pretrained_name = model_cfg.get("pretrained_name", "facebook/wav2vec2-large-xlsr-53")

    def build_processor_and_model():
        try:
            proc = Wav2Vec2Processor.from_pretrained(
                pretrained_name,
                cache_dir=cache_dir,
                force_download=True,
                resume_download=False,
            )
            mdl = Wav2Vec2ForCTC.from_pretrained(
                pretrained_name,
                cache_dir=cache_dir,
                force_download=True,
                resume_download=False,
            )
            return proc, mdl
        except TypeError:
            # Fallback: manually assemble tokenizer + feature extractor from hub files.
            local_config = hf_hub_download(
                pretrained_name, "config.json", cache_dir=cache_dir, force_download=True
            )
            model_dir = Path(local_config).parent
            try:
                vocab_path = hf_hub_download(
                    pretrained_name, "vocab.json", cache_dir=cache_dir, force_download=True
                )
                tokenizer_kwargs = {"vocab_file": vocab_path}
            except EntryNotFoundError:
                tokenizer_file = hf_hub_download(
                    pretrained_name, "tokenizer.json", cache_dir=cache_dir, force_download=True
                )
                tokenizer_kwargs = {"tokenizer_file": tokenizer_file}
            tok_cfg_path = hf_hub_download(
                pretrained_name,
                "tokenizer_config.json",
                cache_dir=cache_dir,
                force_download=True,
            )
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            pad_token = tok_cfg.get("pad_token", "<pad>")
            word_delim = tok_cfg.get("word_delimiter_token", "|")
            unk_token = tok_cfg.get("unk_token", "<unk>")

            tokenizer = Wav2Vec2CTCTokenizer(
                pad_token=pad_token,
                word_delimiter_token=word_delim,
                unk_token=unk_token,
                **tokenizer_kwargs,
            )
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
            proc = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            mdl = Wav2Vec2ForCTC.from_pretrained(model_dir)
            return proc, mdl

    processor, model = build_processor_and_model()
    model.config.ctc_zero_infinity = True
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token or processor.tokenizer.unk_token

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = "longest"

        def __call__(self, features: List[Dict]):
            input_features = [{"input_values": f["input_values"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features, padding=self.padding, return_tensors="pt"
                )
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    def prepare_batch(batch):
        # Read raw waveform from file path
        audio_path = batch["audio_path"]
        speech, sr = sf.read(audio_path)  # speech: np.ndarray

        # If stereo, convert to mono
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)

        # Assume cleaned wavs are already at `sample_rate`
        inputs = processor(speech, sampling_rate=sample_rate)
        batch["input_values"] = inputs.input_values[0]

        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcript"]).input_ids

        return batch

    def add_num_samples(batch):
        """Add number of samples so we can filter out too-short audio."""
        audio_path = batch["audio_path"]
        info = sf.info(audio_path)
        batch["num_samples"] = info.frames
        return batch

    # Minimum duration in seconds for CTC stability
    min_duration_seconds = float(train_cfg.get("min_duration_seconds", 0.5))
    min_samples = int(sample_rate * min_duration_seconds)

    train_dataset = None
    eval_dataset = None

    if hf_splits["train"] is not None:
        # Add length info and filter short clips
        train_dataset = hf_splits["train"].map(add_num_samples)
        train_dataset = train_dataset.filter(
            lambda x: x["num_samples"] >= min_samples
        )
        # Prepare model inputs
        train_dataset = train_dataset.map(
            prepare_batch,
            remove_columns=["audio_path", "transcript", "num_samples"],
        )

    if hf_splits["val"] is not None:
        eval_dataset = hf_splits["val"].map(add_num_samples)
        eval_dataset = eval_dataset.filter(
            lambda x: x["num_samples"] >= min_samples
        )
        eval_dataset = eval_dataset.map(
            prepare_batch,
            remove_columns=["audio_path", "transcript", "num_samples"],
        )

    def compute_metrics(eval_pred):
        pred_logits = eval_pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "wer": wer(label_str, pred_str),
            "cer": cer(label_str, pred_str),
        }

    eval_strategy = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(
        output_dir=train_cfg.get("save_dir", "checkpoints/finetune"),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 4)),
        per_device_eval_batch_size=int(train_cfg.get("batch_size", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        num_train_epochs=int(train_cfg.get("epochs", 20)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        evaluation_strategy=eval_strategy,
        save_strategy="steps",
        eval_steps=int(train_cfg.get("logging_steps", 100)),
        save_steps=int(train_cfg.get("logging_steps", 100)),
        logging_steps=int(train_cfg.get("logging_steps", 100)),
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    save_dir = train_cfg.get("save_dir", "checkpoints/finetune")
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
