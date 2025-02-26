import os
import sys
import json
import time
import argparse
import re
import numpy as np
import torch
import torch.distributed as dist

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)

import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report
)

# ------------------------------------------------------------------------
# Helper: Tee class to duplicate stdout to file (similar to finetune_eval.py)
# ------------------------------------------------------------------------
class Tee(object):
    def __init__(self, name, mode="a"):
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# ------------------------------------------------------------------------
# Helper: tokenization
# ------------------------------------------------------------------------
def tokenize_data(example, tokenizer):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # Transformers expects label field to be named "labels"
    tokenized["labels"] = np.array(example["label"], dtype=np.int64)
    return tokenized

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    global_start = time.time()

    # --------------------------------------------------------------------
    # 1. Logging: capture output in a file, similar to finetune_eval.py
    # --------------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    log_file_path = "logs/baseline_eval_output.log"
    sys.stdout = Tee(log_file_path)
    sys.stderr = sys.stdout  # redirect stderr to same file

    print("\n[INFO] Baseline evaluation script started. Logs appended to:", log_file_path)

    # --------------------------------------------------------------------
    # 2. Distributed init
    # --------------------------------------------------------------------
    local_rank = 0
    if torch.cuda.device_count() > 1 and dist.is_available():
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_P2P_DISABLE"] = "0"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_LAUNCH_TIMEOUT"] = "1200"
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if local_rank == 0:
        print(f"[Rank {local_rank}] Using {num_gpus} GPU(s)" if num_gpus > 0 else "No GPU detected! Running on CPU...")

    # --------------------------------------------------------------------
    # 3. Parse arguments (e.g., debug mode)
    # --------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run script in debug mode with fewer samples")
    args = parser.parse_args()
    DEBUG_MODE = args.debug

    # --------------------------------------------------------------------
    # 4. Load HF token (same as in finetune_eval.py)
    # --------------------------------------------------------------------
    access_token = os.getenv("HF_ACCESS_TOKEN", None)
    if access_token is None:
        raise ValueError("Please set the environment variable HF_ACCESS_TOKEN with your Hugging Face token.")

    # --------------------------------------------------------------------
    # 5. Load dataset in the same way as finetune_eval.py
    #
    #    Note: finetune_eval.py uses `stanfordnlp/imdb` instead of `imdb`.
    # --------------------------------------------------------------------
    if local_rank == 0:
        print("[INFO] Loading IMDB dataset...")

    dataset = load_dataset("stanfordnlp/imdb")
    dataset = dataset.shuffle(seed=42)

    if DEBUG_MODE:
        # Use smaller subsets for debugging
        dataset["train"] = dataset["train"].select(range(50))
        dataset["test"]  = dataset["test"].select(range(10))
    else:
        # Mimic the slicing used in finetune_eval.py
        dataset["train"] = dataset["train"].select(range(20000))
        dataset["test"]  = dataset["test"].select(range(5000))

    # Save original texts for example predictions
    original_test_texts = dataset["test"]["text"][:]
    original_test_labels = dataset["test"]["label"][:]

    # --------------------------------------------------------------------
    # 6. Define model configs (match those used in finetune_eval.py)
    #    Do GPT-2 and LLaMA 3.2-1B as an example. Adjust as needed.
    # --------------------------------------------------------------------
    model_configs = {
        "GPT-2": {
            "model_name": "gpt2",
            "torch_dtype": torch.bfloat16,
            "target_modules": ["c_attn", "c_proj"]
        },
        "LLaMA 3.2-1B": {
            "model_name": "meta-llama/Llama-3.2-1B",
            "torch_dtype": torch.bfloat16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
    }

    # --------------------------------------------------------------------
    # 7. Load tokenizers and create tokenized dataset
    #    Store the tokenized dataset on disk so it is only done once
    # --------------------------------------------------------------------
    tokenizers = {}
    for model_name, config in model_configs.items():
        if local_rank == 0:
            print(f"\n[INFO] Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            token=access_token
        )
        # Fix missing pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

        tokenizers[model_name] = tokenizer

    # Actually tokenize the dataset
    tokenized_datasets = {}
    for model_name, tokenizer in tokenizers.items():
        if local_rank == 0:
            print(f"[INFO] Tokenizing dataset for {model_name}...")

        tokenized_path = f"tokenized_datasets/{model_name}.hf"
        if os.path.exists(tokenized_path):
            if local_rank == 0:
                print(f"  - Found cached tokenized dataset: {tokenized_path}")
            tokenized_dataset = load_from_disk(tokenized_path)
        else:
            # Map the dataset to the tokenization function
            tokenized_dataset = dataset.map(
                lambda x: tokenize_data(x, tokenizer),
                batched=True,
                remove_columns=["text"]
            )
            # Keep only the columns needed
            columns_to_keep = ["input_ids", "attention_mask", "labels"]
            tokenized_dataset.set_format(type="torch", columns=columns_to_keep)
            for split in tokenized_dataset.keys():
                tokenized_dataset[split] = tokenized_dataset[split].remove_columns(
                    [col for col in tokenized_dataset[split].column_names if col not in columns_to_keep]
                )
            # Save to disk
            os.makedirs("tokenized_datasets", exist_ok=True)
            tokenized_dataset.save_to_disk(tokenized_path)

        tokenized_datasets[model_name] = tokenized_dataset

    # --------------------------------------------------------------------
    # 8. Load raw (un-finetuned) models
    # --------------------------------------------------------------------
    models = {}
    for model_name, config in model_configs.items():
        if local_rank == 0:
            print(f"\n[INFO] Loading raw baseline model for {model_name}...")

        tokenizer = tokenizers[model_name]
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=2,
            torch_dtype=config.get("torch_dtype", torch.bfloat16),
            token=access_token,
            load_in_8bit=False
        )
        # Ensure pad token is set in the model config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        model.to(local_rank)

        models[model_name] = {
            "model": model,
            "tokenizer": tokenizer
        }

        torch.cuda.empty_cache()

    # --------------------------------------------------------------------
    # 9. Evaluate
    #   Do a direct pass over the test dataset for each model,
    #   compute accuracy, confusion matrix, PR curve, etc.
    # --------------------------------------------------------------------
    if local_rank == 0:
        print("\n[INFO] Starting baseline inference/evaluation...")

    accuracy_metric = evaluate.load("accuracy")

    # Prepare data collator for batching
    #data_collator = DataCollatorWithPadding(return_tensors="pt")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,  # Ensures batch inputs are the same length
        return_tensors="pt"  # Ensures output tensors are PyTorch-compatible
    )

    # For debug, smaller batch; otherwise bigger
    eval_batch_size = 1 if DEBUG_MODE else 8

    results = {}
    for model_name, data in models.items():
        #model = data["model"]
        #tokenizer = data["tokenizer"]

        model = data["model"].to(local_rank)
        tokenizer = data["tokenizer"]
        tokenizer.pad_token = tokenizer.eos_token

        # Also set this in the model config
        model.config.pad_token_id = tokenizer.pad_token_id

        # Only do test set evaluation
        test_data = tokenized_datasets[model_name]["test"]

        # Move to device in each batch
        # Do a simple DataLoader for the test set
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_data,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=1,          # Use 1 worker process per cpu
            pin_memory=True
        )

        # Collect predictions
        all_labels = []
        all_preds  = []
        all_logits = []

        if local_rank == 0:
            print(f"\nEvaluating {model_name} on {len(test_data)} test samples...")

        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                for k,v in batch.items():
                    batch[k] = v.to(local_rank)
                outputs = model(**batch)
                logits  = outputs.logits
                preds   = torch.argmax(logits, dim=-1)
                logits  = outputs.logits.float()   # cast from BF16 -> float32
                preds   = torch.argmax(logits, dim=-1)


                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        # Flatten everything
        all_preds  = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        # Probability for positive
        y_probs = torch.softmax(torch.tensor(all_logits), dim=-1)[:,1].numpy()

        # Accuracy
        #accuracy_value = float(np.mean(all_preds == all_labels))
        # or use evaluate:
        accuracy_value = accuracy_metric.compute(predictions=all_preds, references=all_labels)["accuracy"]

        inference_time = time.time() - start_time

        if local_rank == 0:
            print(f"[RESULT] {model_name} baseline accuracy: {accuracy_value:.4f}")
            print(f"         Inference time: {inference_time:.2f} s")

        # Store results
        results[model_name] = {
            "accuracy": accuracy_value,
            "inference_time": inference_time
        }

        # ----------------------------------------------------------------
        #  Generate evaluation plots (confusion matrix, ROC, PR, etc.)
        #  Just do it on rank 0 to avoid duplications
        # ----------------------------------------------------------------
        if local_rank == 0:
            output_dir = f"baseline_results/{model_name.replace(' ', '_')}"
            os.makedirs(output_dir, exist_ok=True)

            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
            disp.plot(cmap="Blues", values_format='d')
            plt.title(f"{model_name} Confusion Matrix (Baseline)")
            plt.savefig(f"{output_dir}/confusion_matrix.png")
            plt.close()

            # Classification report for more metrics
            class_report = classification_report(all_labels, all_preds, output_dict=True)
            with open(f"{output_dir}/classification_report.json", "w") as f:
                json.dump(class_report, f, indent=4)

            # Precision-Recall
            precision, recall, _ = precision_recall_curve(all_labels, y_probs)
            plt.figure()
            plt.plot(recall, precision, marker='.', label=f"{model_name}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{model_name} Precision-Recall Curve (Baseline)")
            plt.legend()
            plt.savefig(f"{output_dir}/precision_recall_curve.png")
            plt.close()

            # ROC
            fpr, tpr, _ = roc_curve(all_labels, y_probs)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, marker='.', label=f"{model_name} (AUC={roc_auc:.3f})")
            plt.plot([0,1],[0,1], linestyle='--', label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} ROC Curve (Baseline)")
            plt.legend()
            plt.savefig(f"{output_dir}/roc_curve.png")
            plt.close()

            # Simple distribution of predictions
            plt.figure()
            plt.hist(all_preds, bins=2, edgecolor="black", alpha=0.7)
            plt.xticks([0, 1], ["Negative", "Positive"])
            plt.title(f"{model_name} Prediction Distribution (Baseline)")
            plt.savefig(f"{output_dir}/prediction_distribution.png")
            plt.close()

            # WordCloud for misclassified reviews
            misclassified_indices = [i for i in range(len(all_preds)) if all_preds[i] != all_labels[i]]
            if misclassified_indices:
                mis_texts = [original_test_texts[i] for i in misclassified_indices]
                misjoined = " ".join(mis_texts)
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(misjoined)
                plt.figure(figsize=(10,5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"{model_name} Misclassified Reviews WordCloud (Baseline)")
                plt.savefig(f"{output_dir}/misclassified_wordcloud.png")
                plt.close()

            # Print a few example predictions
            print("\n--- Example Test Reviews and Predictions ---")
            num_examples = min(5, len(test_data))
            for i in range(num_examples):
                example_text = original_test_texts[i]
                example_label = original_test_labels[i]
                # Re-infer quickly
                inputs = tokenizer(
                    example_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                ).to(local_rank)

                with torch.no_grad():
                    out = model(**inputs)
                    pred_label = torch.argmax(out.logits, dim=-1).cpu().numpy()[0]

                print(f"\nReview {i+1}: {example_text}")
                print(f"  Actual Sentiment: {'Positive' if example_label == 1 else 'Negative'}")
                print(f"  {model_name} Prediction: {'Positive' if pred_label == 1 else 'Negative'}")

            print("\n[INFO] Evaluation complete.")

            print(f"[INFO] Cleaning up memory for {model_name}...")

        del model
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------
    # Save overall results
    # --------------------------------------------------------------------
    if local_rank == 0:
        os.makedirs("baseline_results", exist_ok=True)
        results_path = "baseline_results/baseline_eval.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n[INFO] Baseline results saved to {results_path}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"[INFO] Baseline eval script finished in {time.time() - global_start:.2f} seconds.")

# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
