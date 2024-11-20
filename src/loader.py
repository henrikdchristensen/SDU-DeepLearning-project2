import os
import pandas as pd
from datasets import load_dataset

def loader(labels_csv_path, train_csv_path, val_csv_path, test_csv_path):
    # Check if the files exist; if not, load from the remote
    # Check if the files exist; if not, load from the remote source
    if not (os.path.exists(train_csv_path) and os.path.exists(val_csv_path) and os.path.exists(test_csv_path) and os.path.exists(labels_csv_path)):
        print("Data files not found. Loading dataset from remote source...")
        
        os.makedirs("data", exist_ok=True)
        
        # Load the dataset from Hugging Face
        ds = load_dataset("dair-ai/emotion", "split")
        
        # Save train data
        train_data = {
            "text": ds["train"]["text"],
            "label": ds["train"]["label"]
        }
        pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
        
        # Save validation data
        val_data = {
            "text": ds["validation"]["text"],
            "label": ds["validation"]["label"]
        }
        pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
        
        # Save test data
        test_data = {
            "text": ds["test"]["text"],
            "label": ds["test"]["label"]
        }
        pd.DataFrame(test_data).to_csv(test_csv_path, index=False)
        
        # Save labels
        labels_data = {"Index": range(len(ds["train"].features["label"].names)),
                "Labels": ds["train"].features["label"].names}
        pd.DataFrame(labels_data).to_csv(labels_csv_path, index=False)
    
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    labels_df = pd.read_csv(labels_csv_path)
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "labels": labels_df
    }