import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

def compute_metrics(true_labels, predicted_labels, labels):
    accuracy = round(accuracy_score(true_labels, predicted_labels), 4)
    f1 = round(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0), 4)
    precision = round(precision_score(true_labels, predicted_labels, average="weighted", zero_division=0), 4)
    recall = round(recall_score(true_labels, predicted_labels, average="weighted", zero_division=0), 4)
    class_report = classification_report(true_labels, predicted_labels, target_names=labels, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "class_report": class_report,
        "conf_matrix": conf_matrix.tolist()
    }

def print_metrics(metrics):
    for key, value in metrics.items():
        if key == "conf_matrix":
            print("confusion matrix:")
            for row in value:
                print(row)
            print()
        elif key == "class_report":
            print("classification report:")
            print(value)
        else:
            print(f"{key}: {value}")

def save_metrics(label, metrics):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{label}_metrics.txt", "w") as f:
        f.write(f"Accuracy Score: {metrics['accuracy']}\n")
        f.write(f"F1-Score: {metrics['f1']}\n")
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Recall: {metrics['recall']}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['class_report'] + "\n")
        f.write("Confusion Matrix:\n")
        for row in metrics['conf_matrix']:
            f.write(str(row) + "\n")