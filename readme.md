# Javanese ASR Pipeline

This repo trains two ASR baselines (Conformer CTC from scratch and Wav2Vec2 fine-tune) on your own data.

## Prerequisites
- Python 3.10+
- FFmpeg available on PATH (needed for audio conversion).

## Project Layout
- data/ (you provide)
  - wav/ (raw audio) or audio files directly under data/ (supports mislabeled containers like 3gp/mp4).
  - transcripts.xlsx or transcript.csv (text).
- data/wav_clean/ and data/transcript_clean.csv are created by the cleaning step.
- configs/config_scratch.yaml and configs/config_finetune.yaml control training.
- hf_cache/ is a local cache for Hugging Face models (auto-created).
- results/ contains evaluation outputs and training plots.
- checkpoints/ contains saved model weights.
- src/ contains the source code modules.
- scripts/ contains the executable scripts.

## Setup
bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .


## Step 0: Place Your Data
1) Put all audio files in data/wav/ (or directly in data/ if no subfolder).  
2) Put transcripts in data/transcripts.xlsx (sheet “Data” with column Transcript) or data/transcript.csv.

## Step 1: Clean Data
Creates 16 kHz mono WAVs and cleaned transcripts.
bash
python scripts/clean_data.py --transcript_path data/transcripts.xlsx --audio_dir data

Outputs:
- data/wav_clean/ (resampled audio)
- data/transcript_clean.csv (lowercased, punctuation removed except apostrophes)

## Step 2: Train Conformer From Scratch
Adjust configs/config_scratch.yaml (e.g., epochs, model size). Then run:
bash
python scripts/train_scratch.py

Outputs:
- Checkpoint saved to checkpoints/scratch/best_model.pth (best val WER).
- Training plot saved to results/scratch_training_plot.png.

## Step 3: Fine-Tune Wav2Vec2
Adjust configs/config_finetune.yaml (e.g., epochs). Then run:
bash
python scripts/train_finetune.py --cache_dir hf_cache

This downloads facebook/wav2vec2-large-xlsr-53 into hf_cache/ and saves fine-tuned weights in checkpoints/finetune/.

## Step 4: Evaluate
Compares scratch vs fine-tuned on the held-out test split and writes a CSV of predictions.
bash
python scripts/evaluate.py

Outputs:
- Metrics printed (WER/CER for both models).
- results/evaluation_results.csv with filename, target, prediction_scratch, prediction_finetune, and detailed error analysis.

## Notes
- If your transcript file uses different column names or sheets, use the cleaning flags: --text_column, --filename_column, --sheet_name.
- If you encounter download/cache issues, delete hf_cache/ and rerun step 3. You can also set a different cache via --cache_dir path.
- Training on CPU is slow; a GPU build of PyTorch + torchaudio is recommended if available.
