import os
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

def decode_tokens(token_ids, reverse_vocab):
    words = [reverse_vocab[token] for token in token_ids if token in reverse_vocab]
    return " ".join(words)

def predict(label, model, device, loader, label_map, reverse_vocab):
    model.to(device)
    model.eval()

    # Store predictions
    predictions = []
    true_labels = []
    predicted_labels = []

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            sentences, labels = batch
            labels = labels.to(device)

            # Forward pass
            outputs = model(sentences.to(device))
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

            # Collect true and predicted labels for metrics
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Convert tokenized tensors back to text using decode_tokens()
            decoded_sentences = [
                decode_tokens(sentence.tolist(), reverse_vocab)
                for sentence in sentences
            ]

            for i, decoded_sentence in enumerate(decoded_sentences):
                true_label = label_map[labels[i].item()]
                pred_label = label_map[predicted[i].item()]
                confidence = probabilities[i][predicted[i].item()].item()
                correct = true_label == pred_label

                predictions.append({
                    "Sentence": decoded_sentence,
                    "Correct": correct,
                    "True Label": true_label,
                    "Predicted Label": pred_label,
                    "Confidence (%)": f"{confidence * 100:.2f}"
                })

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions)
    predictions_file = f"results/{label}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)

    # Compute evaluation metrics using sklearn
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    class_report = classification_report(
        true_labels, predicted_labels, target_names=list(label_map.values())
    )
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Store all metrics in a single string
    metrics_summary = (
        f"Accuracy: {accuracy:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n\n"
        "Classification Report:\n"
        f"{class_report}\n\n"
        "Confusion Matrix:\n"
        f"{conf_matrix}"
    )

    # Save metrics summary to a single file
    metrics_file = f"results/{label}_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(metrics_summary)