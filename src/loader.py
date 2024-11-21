import os
import pandas as pd
from datasets import load_dataset

def loader(train_csv_path, val_csv_path, test_csv_path):
    # Check if the files exist; if not, load from the remote source
    if not (os.path.exists(train_csv_path) and os.path.exists(val_csv_path) and os.path.exists(test_csv_path)):
        print("Data files not found. Loading dataset from remote source...")
        
        os.makedirs("data", exist_ok=True)
        
        # Load the dataset from Hugging Face
        ds = load_dataset("dair-ai/emotion", "split")
        
        label_names = ds["train"].features["label"].names
        
        # Save train data
        train_data = {
            "text": ds["train"]["text"],
            "label": ds["train"]["label"],
            # Convert label indices to label names
            "label_name": [label_names[label] for label in ds["train"]["label"]]
        }
        pd.DataFrame(train_data).to_csv(train_csv_path, index=True)
        
        # Save validation data
        val_data = {
            "text": ds["validation"]["text"],
            "label": ds["validation"]["label"],
            "label_name": [label_names[label] for label in ds["validation"]["label"]]
        }
        pd.DataFrame(val_data).to_csv(val_csv_path, index=True)
        
        # Save test data
        test_data = {
            "text": ds["test"]["text"],
            "label": ds["test"]["label"],
            "label_name": [label_names[label] for label in ds["test"]["label"]]
        }
        pd.DataFrame(test_data).to_csv(test_csv_path, index=True)
    
    train_df = pd.read_csv(train_csv_path, index_col=0)
    val_df = pd.read_csv(val_csv_path, index_col=0)
    test_df = pd.read_csv(test_csv_path, index_col=0)
    
    return train_df, val_df, test_df