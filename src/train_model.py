import os
import time
import numpy as np
from collections import Counter

from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary

from metrics import compute_metrics, print_metrics, save_metrics

def log_undefined(predicted_labels, labels):
    counts = Counter(predicted_labels)
    for idx, label in enumerate(labels):
        if counts[idx] == 0:
            print(f"Warning: No predictions for label '{label}' (index {idx}).")


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=1)  # get index of highest score (predicted label)
    correct = (predicted == targets).sum().item()  # count number of correct predictions
    return correct

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
    if step_size is not None and gamma is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Metrics storage
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    total_start_time = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        total_train_loss, train_correct, total_train_samples = 0.0, 0, 0

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

            total_train_loss += loss.item() * inputs.size(0)
            train_correct += calculate_accuracy(outputs, targets)
            total_train_samples += inputs.size(0)

        # Update learning rate scheduler
        if step_size and gamma:
            scheduler.step()

        # Record training metrics
        train_loss = total_train_loss / total_train_samples
        train_accuracy = 100 * train_correct / total_train_samples
        train_losses.append(round(train_loss, 4))
        train_accuracies.append(round(train_accuracy, 2))

        # Validation phase
        model.eval()
        total_val_loss, val_correct, total_val_samples = 0.0, 0, 0

        # Store true labels and predictions for the entire validation set
        all_true_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, dim=1)

                # Append true and predicted labels
                all_true_labels.extend(targets.cpu().numpy())
                all_predicted_labels.extend(predicted.cpu().numpy())
                
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item() * inputs.size(0)
                val_correct += calculate_accuracy(outputs, targets)
                total_val_samples += inputs.size(0)

        # Record validation metrics
        val_loss = total_val_loss / total_val_samples
        val_accuracy = 100 * val_correct / total_val_samples
        val_losses.append(round(val_loss, 4))
        val_accuracies.append(round(val_accuracy, 2))

        epoch_duration = round(time.time() - start_time)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.3f} (acc. {train_accuracies[-1]:.1f}%) | "
            f"Val Loss: {val_losses[-1]:.3f} (acc. {val_accuracies[-1]:.1f}%) | Time: {epoch_duration}s"
        )

    # Log undefined predictions
    log_undefined(all_predicted_labels, label_map.values())
    
    # Total training time
    total_training_time = round(time.time() - total_start_time)
    print(f"Total Training Time: {total_training_time}s\n")
    
    # Save model summary and metrics
    os.makedirs("models", exist_ok=True)
    with open(f"models/{label}.txt", "w") as f:
        f.write(str(summary(model, verbose=0)))
    
    # Convert to NumPy arrays
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)

    # Compute metrics
    metrics = compute_metrics(all_true_labels, all_predicted_labels, label_map.values())
    print_metrics(metrics)
    save_metrics(label, metrics)

    # Save model state
    torch.save(model.state_dict(), f"models/{label}.pth")

    # Return training history
    return {
        "num_epochs": num_epochs,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }
