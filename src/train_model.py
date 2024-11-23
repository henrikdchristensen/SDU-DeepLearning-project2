import os
import time
import numpy as np
from collections import Counter

from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary

from sklearn.metrics import accuracy_score, f1_score

from metrics import compute_metrics, print_metrics, save_metrics

def log_undefined(predicted_labels, labels):
    counts = Counter(predicted_labels)
    for idx, label in enumerate(labels):
        if counts[idx] == 0:
            print(f"Warning: No predictions for label '{label}' (index {idx}).")


import os
import time
import numpy as np
from collections import Counter
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from sklearn.metrics import accuracy_score, f1_score
from metrics import compute_metrics, print_metrics, save_metrics


def log_undefined(predicted_labels, labels):
    """Logs warnings for labels that are not predicted."""
    counts = Counter(predicted_labels)
    for idx, label in enumerate(labels):
        if counts[idx] == 0:
            print(f"Warning: No predictions for label '{label}' (index {idx}).")


def train_model(
    label,
    model,
    train_loader,
    val_loader,
    label_map,
    device,
    optimizer_type="Adam",
    learning_rate=0.001,
    momentum=0.9,
    weight_decay=0.0,
    step_size=None,
    gamma=0.5,
    reg_type=None,
    reg_lambda=0.0,
    num_epochs=30
):
    """Trains a PyTorch model and logs metrics for each epoch."""
    # Move the model to the device
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Learning rate scheduler
    scheduler = None
    if step_size is not None and gamma is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Metrics storage
    train_losses, train_accuracies, train_f1_scores = [], [], []
    val_losses, val_accuracies, val_f1_scores = [], [], []

    total_start_time = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        
        epoch_total_train_loss = 0.0
        epoch_total_train_samples = 0
        epoch_train_true_labels = []
        epoch_train_predicted_labels = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Apply regularization
            if reg_type == "L1":
                l1_norm = sum(param.abs().sum() for param in model.parameters())
                loss += reg_lambda * l1_norm
            elif reg_type == "L2":
                l2_norm = sum(param.pow(2).sum() for param in model.parameters())
                loss += reg_lambda * l2_norm

            loss.backward()
            optimizer.step()

            epoch_total_train_loss += loss.item() * inputs.size(0)
            epoch_total_train_samples += inputs.size(0)

            # Collect true labels and predictions
            _, predicted = torch.max(outputs, dim=1)
            epoch_train_true_labels.extend(targets.cpu().numpy())
            epoch_train_predicted_labels.extend(predicted.cpu().numpy())

        # Calculate training metrics
        avg_epoch_train_loss = round(epoch_total_train_loss / epoch_total_train_samples, 4)
        epoch_train_accuracy = round(accuracy_score(epoch_train_true_labels, epoch_train_predicted_labels), 4)
        epoch_train_f1 = round(f1_score(epoch_train_true_labels, epoch_train_predicted_labels, average='weighted'), 4)
        train_losses.append(avg_epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        train_f1_scores.append(epoch_train_f1)

        # Validation phase
        model.eval()
        
        epoch_total_val_loss = 0.0
        epoch_total_val_samples = 0
        all_val_true_labels = []
        all_val_predicted_labels = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                avg_epoch_val_loss = criterion(outputs, targets)
                epoch_total_val_loss += avg_epoch_val_loss.item() * inputs.size(0)
                epoch_total_val_samples += inputs.size(0)

                # Collect true labels and predictions
                _, predicted = torch.max(outputs, dim=1)
                all_val_true_labels.extend(targets.cpu().numpy())
                all_val_predicted_labels.extend(predicted.cpu().numpy())

        # Calculate validation metrics
        avg_epoch_val_loss = round(epoch_total_val_loss / epoch_total_val_samples, 4)
        epoch_val_accuracy = round(accuracy_score(all_val_true_labels, all_val_predicted_labels), 4)
        epoch_val_f1 = round(f1_score(all_val_true_labels, all_val_predicted_labels, average='weighted'), 4)
        val_losses.append(avg_epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        val_f1_scores.append(epoch_val_f1)

        # Update learning rate
        if scheduler:
            scheduler.step()

        epoch_duration = round(time.time() - start_time)
        print(
            f"Epoch {epoch + 1}/{num_epochs} ({epoch_duration}s) | "
            f"Train: loss {avg_epoch_train_loss}, acc {epoch_train_accuracy*100:.2f}%, f1 {epoch_train_f1*100:.2f}% | "
            f"Val: loss {avg_epoch_val_loss}, acc {epoch_val_accuracy*100:.2f}%, f1 {epoch_val_f1*100:.2f}%"
        )

    # Log undefined predictions
    log_undefined(all_val_predicted_labels, label_map.values())

    # Total training time
    total_training_time = round(time.time() - total_start_time)
    print(f"Total Training Time: {total_training_time}s\n")

    # Save model summary and metrics
    os.makedirs("models", exist_ok=True)
    with open(f"models/{label}.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(model, verbose=0)))

    # Compute and save metrics
    metrics = compute_metrics(all_val_true_labels, all_val_predicted_labels, label_map.values())
    print_metrics(metrics)
    save_metrics(label, metrics)

    # Save model state
    torch.save(model.state_dict(), f"models/{label}.pth")

    # Return training history
    return {
        "num_epochs": num_epochs,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "train_f1_scores": train_f1_scores,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_f1_scores": val_f1_scores,
    }

