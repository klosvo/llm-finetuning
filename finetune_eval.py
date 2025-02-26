import argparse
import json
import os
import sys
import time
import re

import torch
import torch.distributed as dist
import torch._dynamo
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback
)
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import evaluate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report
)

from wordcloud import WordCloud
from concurrent.futures import ThreadPoolExecutor

class TimeTrackingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.start_time = None
        self.step_times = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.start_time
        self.step_times.append(step_time)

    def on_train_end(self, args, state, control, **kwargs):
        # Save the step times to a JSON file in output_dir
        save_path = os.path.join(self.output_dir, "step_times.json")
        with open(save_path, "w") as f:
            json.dump(self.step_times, f, indent=4)
        print(f"Step times saved to {save_path}")

# Redirect stdout and stderr to both console and log file
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

def format_dataset(example):
    # For input_ids and attention_mask, use clone().detach() if they are already tensors.
    def safe_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach().to(torch.long)
        return torch.tensor(x, dtype=torch.long)

    return {
        "input_ids": safe_tensor(example["input_ids"]),
        "attention_mask": safe_tensor(example["attention_mask"]),
        "labels": example["labels"]
    }

# Tokenization function
def tokenize_data(example, tokenizer):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",  # Ensures uniform length
        truncation=True,  # Prevents excessive nesting
        max_length=512,  # Controls input size
        return_tensors="pt"  # Ensures correct format before converting to PyTorch tensors
    )
    tokenized["labels"] = np.array(example["label"], dtype=np.int64)  # Ensure scalar labels
    return tokenized

def gather_data_across_gpus(data):
    """Gathers data from all GPUs."""
    gathered_data = [None] * dist.get_world_size()  # Create a list to store data from all GPUs
    dist.all_gather_object(gathered_data, data)  # Gather data across ranks
    return [item for sublist in gathered_data for item in sublist]  # Flatten the list

def extract_checkpoint_step(ckpt_name):
    """
	Extracts the checkpoint step from a directory name using regex.
    Returns -1 if the pattern is not found.
    """
    match = re.search(r"checkpoint-(\d+)", ckpt_name)
    if match:
        return int(match.group(1))
    else:
        return -1

def compute_accuracy(predictions, references):
    return np.mean(predictions == references)


def main():
    global_start = time.time()  # Track total training duration

    # Ensure logs are appended instead of overwritten
    log_file_path = "logs/training_output.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Open log file in append mode
    log_file = open(log_file_path, "a")

    sys.stdout = Tee(log_file_path)
    sys.stderr = sys.stdout  # Redirect stderr too

    print("\nTraining script started. Logs will be appended to:", log_file_path)


    # Initialize distributed processing
    local_rank = 0  # Default to 0 if not using distributed training
    if torch.cuda.device_count() > 1 and dist.is_available():
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_P2P_DISABLE"] = "0"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Prevents NCCL deadlocks
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # Debugging info
        os.environ["NCCL_BLOCKING_WAIT"] = "1"         # Forces blocking mode
        os.environ["NCCL_LAUNCH_TIMEOUT"] = "1200"      # Increase launch timeout (in seconds)
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Load Hugging Face access token
    access_token = os.getenv("HF_ACCESS_TOKEN", None)
    if access_token is None:
        raise ValueError("Please set the Hugging Face access token as an environment variable: HF_ACCESS_TOKEN")

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if local_rank == 0:
        print(f"[Rank {local_rank}] Using {num_gpus} GPU(s)" if num_gpus > 0 else "No GPU detected! Running on CPU...")
    
    num_workers = os.cpu_count()  # or a fixed default like 4

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run script in debug mode with reduced dataset and steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory")
    args = parser.parse_args()

    DEBUG_MODE = args.debug

    # Load dataset
    if local_rank == 0:
        print("[INFO] Loading IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb")
    dataset = dataset.shuffle(seed=42)

    if DEBUG_MODE:
        dataset["train"] = dataset["train"].select(range(50))
        dataset["test"] = dataset["test"].select(range(10))
    else:
        dataset["train"] = dataset["train"].select(range(20000))
        dataset["test"] = dataset["test"].select(range(5000))

    # Define model configurations
    model_configs = {
#        "GPT-2": {
#            "model_name": "gpt2",
#            "torch_dtype": torch.bfloat16,
#            "target_modules": ["c_attn", "c_proj"],
#        },
        "LLaMA 3.2-1B": {
            "model_name": "meta-llama/Llama-3.2-1B",
            "torch_dtype": torch.bfloat16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
    }

    models = {}
    tokenizers = {}

    for model_name, config in model_configs.items():
        tokenizers[model_name] = AutoTokenizer.from_pretrained(config["model_name"], token=access_token)
        
        # Set a pad_token if it's missing
        if tokenizers[model_name].pad_token is None:
            tokenizers[model_name].pad_token = tokenizers[model_name].eos_token  # Use EOS as padding token
            tokenizers[model_name].padding_side = "right"  # Right padding is preferred

    # Tokenize dataset for each model separately
    tokenized_datasets = {}

    for model_name, tokenizer in tokenizers.items():
        if local_rank == 0:
            print(f"[INFO] Tokenizing dataset for {model_name}...")

        tokenized_path = f"tokenized_datasets/{model_name}.hf"

        if os.path.exists(tokenized_path):  
            if local_rank == 0:
                print(f"[INFO] Loading pre-tokenized dataset from {tokenized_path}")
            tokenized_dataset = load_from_disk(tokenized_path)  
        else:
            tokenized_dataset = dataset.map(
                lambda x: tokenize_data(x, tokenizer), 
                batched=True, 
                remove_columns=["text"] 
            )

            # Ensure only required columns remain
            columns_to_keep = ["input_ids", "attention_mask", "labels"]
            tokenized_dataset.set_format(type="torch", columns=columns_to_keep)

            # Iterate over each split (train, test, unsupervised) and remove columns individually
            for split in tokenized_dataset.keys():
                tokenized_dataset[split] = tokenized_dataset[split].remove_columns([
                    col for col in tokenized_dataset[split].column_names if col not in columns_to_keep
                ])

            # Apply mapping function to each dataset separately
            for split in tokenized_dataset.keys():
                tokenized_dataset[split] = tokenized_dataset[split].map(format_dataset, batched=False)

            tokenized_dataset.save_to_disk(tokenized_path)

        tokenized_datasets[model_name] = tokenized_dataset  # Store back the processed dataset


    # Store original test dataset before modifying it
    original_test_texts = dataset["test"]["text"][:]  
    original_test_labels = dataset["test"]["label"][:]  

    # Load models dynamically
    for model_name, config in model_configs.items():
        if local_rank == 0:
            print(f"\n[INFO] Loading {model_name} model...")

        tokenizer = tokenizers[model_name]
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=2,
            torch_dtype=config.get("torch_dtype", torch.bfloat16),
            token=access_token,
            load_in_8bit=False
        )

        # Set pad_token_id on the model
        model.config.pad_token_id = tokenizer.pad_token_id

        # Disable gradient checkpointing (fixes LoRA conflict)
        model.config.use_cache = False
        model.gradient_checkpointing_enable = False

        model = model.to(local_rank)

        # Apply LoRA for memory-efficient fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=config["target_modules"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Right padding is preferred

        models[model_name] = {"tokenizer": tokenizer, "model": model}
        torch.cuda.empty_cache()

    if local_rank == 0:
        print("[INFO] Models loaded successfully.")

    # Load accuracy metric
    acc_metric = evaluate.load("accuracy")

    # Training loop
    for model_name, model_data in models.items():

        model = model_data["model"].to(local_rank)
        tokenizer = model_data["tokenizer"]
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        if DEBUG_MODE:
            per_device_train_batch_size = 1
            per_device_eval_batch_size = 1
        else:
            per_device_train_batch_size = 8
            per_device_eval_batch_size = 8

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,  # Ensures batch inputs are the same length
            return_tensors="pt"  # Ensures output tensors are PyTorch-compatible
        )

        train_sampler = DistributedSampler(
            tokenized_datasets[model_name]["train"],
            num_replicas=torch.distributed.get_world_size(),
            rank=local_rank,
            shuffle=True,
            drop_last=True
        )

        train_dataloader = DataLoader(
            tokenized_datasets[model_name]["train"],
            batch_size=per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=1,          # Use 1 worker process per cpu
            pin_memory=True 
        )

        test_sampler = DistributedSampler(
            tokenized_datasets[model_name]["test"],
            num_replicas=torch.distributed.get_world_size(),
            rank=local_rank,
            shuffle=False
        )

        test_dataloader = DataLoader(
            tokenized_datasets[model_name]["test"],
            batch_size=per_device_eval_batch_size,  # Use the evaluation batch size
            sampler=test_sampler,
            collate_fn=data_collator,
            num_workers=1,          # Use 1 worker processes per cpu
            pin_memory=True 
        )

        # Configure debug mode
        if DEBUG_MODE:
            train_data = tokenized_datasets[model_name]["train"].select(range(5))
            test_data = tokenized_datasets[model_name]["test"].select(range(5))
            max_steps = 1
            per_device_train_batch_size = 1
            per_device_eval_batch_size = 1
            save_steps = 1
            eval_steps=1
            num_train_epochs = 1
            output_dir = f"debug/checkpoints/{model_name.replace(' ', '_')}"
        else:
            train_data = tokenized_datasets[model_name]["train"]
            test_data = tokenized_datasets[model_name]["test"]
            max_steps = -1
            per_device_train_batch_size = 8
            per_device_eval_batch_size = 8
            eval_steps=20
            save_steps = 500
            num_train_epochs = 5
            output_dir = f"checkpoints/{model_name.replace(' ', '_')}"

        if local_rank == 0:
            print(f"[INFO] Dataset sizes for {model_name} - Train: {len(train_data)}, Test: {len(test_data)}")
            print(f"\n[INFO] Training {model_name} on {num_gpus} GPU(s)...")
            print(f"[INFO] Training settings: {num_train_epochs} epochs, batch size {per_device_train_batch_size}, saving every {save_steps} steps.")

        os.makedirs(output_dir, exist_ok=True)

        # List only those directories that start with "checkpoint-"
        checkpoint_dirs = [ckpt for ckpt in os.listdir(output_dir) if ckpt.startswith("checkpoint-")]

        # Filter out any directories that did not match the expected pattern (step == -1)
        valid_checkpoints = [ckpt for ckpt in checkpoint_dirs if extract_checkpoint_step(ckpt) != -1]

        if args.resume_from_checkpoint or valid_checkpoints:
            # Sort the checkpoints by the extracted step number
            valid_checkpoints = sorted(valid_checkpoints, key=lambda x: extract_checkpoint_step(x))
            last_checkpoint = os.path.join(output_dir, valid_checkpoints[-1])
            if local_rank == 0:
                print(f"Resuming from checkpoint: {last_checkpoint}")
        else:
            last_checkpoint = None
            print("No valid checkpoints found; starting training from scratch.")

        os.makedirs(f"{output_dir}/logs", exist_ok=True)  # Ensures logging dir exists

        training_args = TrainingArguments(
            optim="adamw_torch",
            output_dir=output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=2,  # Keep only last 2 checkpoints
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=5 if DEBUG_MODE else 20,
            fp16=True,
            report_to="none",
            dataloader_drop_last=False,  # Don't drop last batch (prevents padding issues)
            label_names=["labels"],

            # Parameters for early stopping
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,  # since higher accuracy is better
            load_best_model_at_end=True,
            save_on_each_node=True
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = acc_metric.compute(predictions=predictions, references=labels)
            # Ensure the returned dictionary key matches what was set in TrainingArguments.
            return {"eval_accuracy": acc["accuracy"]}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Add TimeTrackingCallback before training starts
        callback = TimeTrackingCallback(output_dir=output_dir)
        trainer.add_callback(callback)

        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=15))  # Stops if no improvement after 15 evals

        model = torch.compile(model)

        if local_rank == 0:
            print("\nStarting Training for 1 Epoch...")

        training_start = time.time()  # Track total training duration

        # If a checkpoint exists, resume training
        if last_checkpoint:
            if local_rank == 0:
                print(f"[INFO] Resuming training from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)  # Normal Trainer resume
        else:
            trainer.train()  # Start training from scratch

        training_duration = time.time() - training_start

        if local_rank == 0:  # Only rank 0 saves checkpoints (avoids overwrites)
            print(f"Training completed in {time.time() - training_start:.2f} seconds.\n")
            print(f"[INFO] Saving final model for {model_name}...")
            # After training completes, save the final model and tokenizer in one pass
            final_model_dir = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            print(f"Final model saved to {final_model_dir}")

            print(f"[INFO] Evaluating {model_name}...")

        # Get predictions from the trainer.
        predictions = trainer.predict(test_data)

        if local_rank == 0:
            # On rank 0, predictions.predictions is already the full, aggregated output.
            y_pred = torch.argmax(torch.tensor(predictions.predictions), dim=-1).tolist()
            # Use test_data["labels"] directly, since on rank 0 it should contain the full set.
            y_true = test_data["labels"]
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.tolist()
        
            # Compute metrics asynchronously
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    acc_metric.compute,
                    predictions=np.array(y_pred),
                    references=np.array(y_true)
                )
                eval_results = future.result()  # This will block until the computation is done

            print("Evaluation Results:", eval_results)
            print(f"[RESULT] {model_name} Accuracy: {eval_results['accuracy']:.4f}")

            with open(f"{output_dir}/eval_results.json", "w") as f:
                json.dump(eval_results, f, indent=4)

            results = {
                "training_time": training_duration,
                "model_name": model_name,
                "final_accuracy": eval_results["accuracy"],
                "step_times": callback.step_times,
            }

            # Save to JSON
            with open(f"{output_dir}/training_results.json", "w") as f:
                json.dump(results, f, indent=4)
            print(f"[INFO] Training results saved to {output_dir}/training_results.json")


        # Print five example reviews and model predictions
        if local_rank == 0:
            print("\n--- Example IMDB Reviews and Model Predictions ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(5):
            example_text = original_test_texts[i]
            example_label = original_test_labels[i]

            inputs = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs = {key: val.to(device) for key, val in inputs.items() if isinstance(val, torch.Tensor)}

            # Disable TorchInductor before inference
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 0  # Disable caching issues

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                example_prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]

            if local_rank == 0:
                print(f"\nReview {i+1}: {example_text}")
                print(f"Actual Sentiment: {'Positive' if example_label == 1 else 'Negative'}")
                print(f"{model_name} Predicted Sentiment: {'Positive' if example_prediction == 1 else 'Negative'}")

        train_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        eval_loss = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]

        # Gather probability scores (y_probs) for PR and ROC curves
        if not isinstance(predictions.predictions, torch.Tensor):
            logits_local = torch.tensor(predictions.predictions)
        else:
            logits_local = predictions.predictions

        if logits_local.dim() > 1 and logits_local.shape[-1] == 2:  
            y_probs = torch.nn.functional.softmax(logits_local, dim=-1)[:, 1].numpy()
        else:
            y_probs = logits_local.numpy()

        step_times_all = callback.step_times

        if local_rank == 0:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_probs = np.array(y_probs)
            train_loss = np.array(train_loss)
            eval_loss = np.array(eval_loss)

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            # Ensure both labels (0 and 1) are included in the confusion matrix
            labels = [0, 1]  # ["Negative", "Positive"]

            # Force the confusion matrix to include both labels, even if one is missing
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
            disp.plot(cmap="Blues", values_format='d')

            # Manually set tick labels to match the expected labels
            disp.ax_.set_xticks([0, 1])
            disp.ax_.set_xticklabels(["Negative", "Positive"])
            disp.ax_.set_yticks([0, 1])
            disp.ax_.set_yticklabels(["Negative", "Positive"])

            plt.title(f"{model_name} Confusion Matrix")
            plt.savefig(f"{output_dir}/confusion_matrix.png")
            plt.close()

            pd.DataFrame(cm).to_csv(f"{output_dir}/confusion_matrix.csv", index=False)

            # Training Loss Plot
            plt.figure()
            plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title(f"{model_name} Training Loss Curve")
            plt.legend()
            plt.savefig(f"{output_dir}/training_loss.png")
            plt.close()

            # Compute precision-recall values
            precision, recall, _ = precision_recall_curve(y_true, y_probs)

            # Plot Precision-Recall Curve
            plt.figure()
            plt.plot(recall, precision, marker='.', label=f"{model_name} PR Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{model_name} Precision-Recall Curve")
            plt.legend()
            plt.savefig(f"{output_dir}/precision_recall_curve.png")
            plt.close()

            # Compute ROC values
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)

            # Plot ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, marker='.', label=f"{model_name} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")  # Baseline
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} ROC Curve")
            plt.legend()
            plt.savefig(f"{output_dir}/roc_curve.png")
            plt.close()


            plt.figure()
            plt.hist(y_pred, bins=2, edgecolor="black", alpha=0.7)
            plt.xticks([0, 1], ["Negative", "Positive"])
            plt.xlabel("Predicted Sentiment")
            plt.ylabel("Count")
            plt.title(f"{model_name} Sentiment Distribution")
            plt.savefig(f"{output_dir}/sentiment_distribution.png")
            plt.close()


            # Get incorrectly classified reviews
            misclassified_texts = [dataset["test"][i]["text"] for i in range(len(y_pred)) if y_pred[i] != y_true[i]]
            misclassified_words = " ".join(misclassified_texts)

            if misclassified_words.strip():
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(misclassified_words)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"{model_name} Misclassified Reviews Word Cloud")
                plt.savefig(f"{output_dir}/misclassified_wordcloud.png")
                plt.close()
            else:
                print("No misclassified reviews to generate a word cloud.")


            plt.figure()
            plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
            plt.plot(range(len(eval_loss)), eval_loss, label="Validation Loss", linestyle="dashed")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title(f"{model_name} Training vs Validation Loss")
            plt.legend()
            plt.savefig(f"{output_dir}/training_vs_validation_loss.png")
            plt.close()


            # Get per-class accuracy
            report = classification_report(y_true, y_pred, output_dict=True)
            class_accuracies = [
                report.get("0", {}).get("precision", 0),
                report.get("1", {}).get("precision", 0)
            ]

            labels = ["Negative", "Positive"]

            # Plot per-class accuracy
            plt.figure()
            sns.barplot(x=labels, y=class_accuracies, palette="Blues")
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title(f"{model_name} Per-Class Accuracy")
            plt.savefig(f"{output_dir}/per_class_accuracy.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.hist(step_times_all, bins=50, edgecolor="black", alpha=0.7)
            plt.xlabel("Step Time (seconds)")
            plt.ylabel("Frequency")
            plt.title(f"{model_name} Step Time Distribution")
            plt.grid()
            plt.savefig(f"{output_dir}/step_time_distribution.png")
            plt.close()

            print("\n[INFO] Evaluation complete. Results saved.")

            print(f"[INFO] Cleaning up memory for {model_name}...")

        del model
        torch.cuda.empty_cache()

    # Destroy NCCL process group
    if dist.is_initialized():
        dist.destroy_process_group()

    print("\nAll models trained and evaluated successfully!")
    total_duration = time.time() - global_start
    print(f"Program completed in {time.time() - global_start:.2f} seconds.\n")

if __name__ == "__main__":
    main()

