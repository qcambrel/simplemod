import json
import torch
import shutil
import warnings
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set the device to use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset class for the model pipeline
class ReviewDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        self.reviews = dataset["text_"].tolist()
        self.labels = dataset["is_fake"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.reviews[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
# Define the metrics function
def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Load the preprocessed dataset
df = pd.read_csv("reviews_ready.csv")

# Load the pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the cross-validation strategy
# No shuffling due to potential data leakage
skf = StratifiedKFold(n_splits=5)

# Define hyperparameter space
params_grid = [
    {
        "learning_rate": 1e-5,
        "batch_size": 16
    }, {
        "learning_rate": 2e-5,
        "batch_size": 32
    }, {
        "learning_rate": 3e-5,
        "batch_size": 32
    }
]

# Define a list to store the fold results
fold_results = []

# Ignore warnings
warnings.filterwarnings("ignore")

# Choose best results if available
try:
    with open("best_params.json", "r") as f:
        best_params = json.load(f)
except FileNotFoundError:
    best_params = None

# Skip cross-validation and hyperparameter tuning if best params are available
if not best_params:
    print("No best params found. Running cross-validation and hyperparameter tuning...")

    # Perform cross-validation using stratified k-fold
    for fold, (train_idx, eval_idx) in enumerate(skf.split(df["text_"], df["is_fake"])):
        print(f"\n===Fold {fold + 1}===")

        # Split the dataset into training and evaluation sets
        train_df = df.iloc[train_idx]
        eval_df = df.iloc[eval_idx]

        # Create the dataset objects
        train_dataset = ReviewDataset(train_df, tokenizer)
        eval_dataset = ReviewDataset(eval_df, tokenizer)

        # Set inital values
        best_score = 0
        best_params = None
        best_result = None

        # Define hyperparameter tuning loop using manual grid search
        for params in params_grid:
            print(f"  ➤ Training with: LR={params['learning_rate']}, BS={params['batch_size']}")

            # Load the pretrained BERT model
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

            # Move the model to the device
            model.to(device)

            # Define the training arguments
            training_args = TrainingArguments(
                output_dir=f"./results/fold_{fold+1}/lr{params['learning_rate']}_bs{params['batch_size']}",
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                logging_dir="./logs",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True
            )

            # Define the trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics
            )

            # Train and evaluate the model
            trainer.train()
            results = trainer.evaluate()

            # Store the results in a JSON file
            with open("results.jsonl", "a") as f:
                json.dump(results, f, indent=4)
                f.write("\n")

            # Delete current fold output directory
            shutil.rmtree(f"./results/fold_{fold + 1}", ignore_errors=True)
        
            print(f"     F1 score: {results['eval_f1']:.4f}")

            # Determine the best model (based on F1 score)
            if results["eval_f1"] > best_score:
                best_score = results["eval_f1"]
                best_params = params
                best_result = results

                # Store the best hyperparameters
                with open("best_params.json", "w") as f:
                    json.dump(best_params, f, indent=4)
                
                # Optional: Save best model per fold
                # trainer.save_model(f"./best_model/fold_{fold}")
                # tokenizer.save_pretrained(f"./best_model/fold_{fold}")

        # Store the best fold results
        fold_results.append(best_result)

        # Optional: Save the trained model and tokenizer
        # trainer.save_model(f"./model/fold_{fold+1}")
        # tokenizer.save_pretrained(f"./tokenizer/fold_{fold+1}")

    # Print the aggregated evaluation results
    final_scores = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
    print(final_scores)

# Print the best hyperparameters
print(f"Best hyperparameters: {best_params}")

# Retrain full model using best hyperparameters
best_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
best_model.to(device)

# Split the dataset into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["is_fake"])

# Create the dataset objects
train_dataset = ReviewDataset(train_df, tokenizer)
eval_dataset = ReviewDataset(train_df, tokenizer)

# Define the training arguments
best_training_args = TrainingArguments(
    output_dir="./results/final",
    num_train_epochs=3,
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Define the trainer
best_trainer = Trainer(
    model=best_model,
    args=best_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
best_trainer.train()

# Evaluate the final model
results = best_trainer.evaluate()

print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall: {results['eval_recall']:.4f}")
print(f"F1: {results['eval_f1']:.4f}")

parser = argparse.ArgumentParser()
parser.add_argument("--upload_model", type=bool, default=False)
parser.add_argument("--save_model", type=bool, default=False)
args = parser.parse_args()

if args.upload_model:
    # Optional: Upload the model and tokenizer to Hugging Face Hub
    best_trainer.push_to_hub("simplemod/final")
    tokenizer.push_to_hub("simplemod/final")

if args.save_model:
    # Optional: Save the trained model and tokenizer
    best_trainer.save_model("./model/final")
    tokenizer.save_pretrained("./tokenizer/final")

# Final results
# Accuracy: 0.9854
# Precision: 0.9722
# Recall: 0.9994
# F1: 0.9856
