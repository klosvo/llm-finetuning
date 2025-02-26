# Fine-Tuning and Baseline Evaluation for IMDB Sentiment Classification

## Overview

This project fine-tunes and evaluates GPT-2 and LLaMA 3.2-1B models for sentiment classification using the IMDB dataset. The project includes two scripts:

- `finetune_eval.py`: Fine-tunes the models with LoRA and evaluates their performance.
- `baseline.py`: Evaluates the raw, pre-trained models without fine-tuning as a baseline comparison.

## Features

- **Distributed Training Support:** Uses `torch.distributed` for multi-GPU training.
- **Hugging Face Integration:** Loads datasets and pre-trained models from Hugging Face.
- **LoRA Fine-Tuning:** Utilizes parameter-efficient training via LoRA in `finetune_eval.py`.
- **Baseline Evaluation:** Assesses unmodified models in `baseline.py`.
- **Metrics & Evaluation:** Computes accuracy, precision-recall, ROC curves, and confusion matrices.
- **Logging & Checkpointing:** Supports logging, checkpoint resumption, and detailed step timing.
- **Data Visualization:** Generates plots for training loss, precision-recall, ROC curves, confusion matrices, and word clouds for misclassified reviews.

## Installation

### Prerequisites

Ensure you have Python 3.8+ and `pip` installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Multi-GPU Training

For multi-GPU training, use `torch.distributed.launch` or `torchrun`. Example:

```bash
torchrun --nproc_per_node=4 finetune_eval.py --debug
```

This command runs the script using 4 GPUs. Adjust `--nproc_per_node` based on the number of available GPUs.

For the baseline evaluation on multiple GPUs:

```bash
torchrun --nproc_per_node=4 baseline.py --debug
```

### Running the Fine-Tuning Script (`finetune_eval.py`)

Execute the script with:

```bash
python finetune_eval.py --debug
```

Options:

- `--debug`: Enables debug mode with a smaller dataset and reduced training steps.
- `--resume_from_checkpoint <path>`: Resume training from a specific checkpoint.

### Running the Baseline Evaluation Script (`baseline.py`)

Execute the baseline evaluation script with:

```bash
python baseline.py --debug
```

Options:

- `--debug`: Runs the evaluation on a smaller dataset for faster testing.

### Environment Variables

Set your Hugging Face API token as an environment variable:

```bash
export HF_ACCESS_TOKEN=<your_token>
```

## Outputs

After running the scripts, the following are saved:

### Fine-Tuning (`finetune_eval.py`)

- **Trained Models & Tokenizers**: Stored in `checkpoints/`.
- **Evaluation Metrics**: Saved as JSON (`eval_results.json`).
- **Training Logs**: Stored in `logs/training_output.log`.
- **Visualization Outputs**:
  - `confusion_matrix.png`
  - `precision_recall_curve.png`
  - `roc_curve.png`
  - `training_loss.png`
  - `sentiment_distribution.png`
  - `misclassified_wordcloud.png`

### Baseline Evaluation (`baseline.py`)

- **Evaluation Metrics**: Saved in `baseline_results/baseline_eval.json`.
- **Logs**: Stored in `logs/baseline_eval_output.log`.
- **Visualization Outputs**:
  - `baseline_results/confusion_matrix.png`
  - `baseline_results/precision_recall_curve.png`
  - `baseline_results/roc_curve.png`
  - `baseline_results/prediction_distribution.png`
  - `baseline_results/misclassified_wordcloud.png`

## License

This project is open-source and free to use.

