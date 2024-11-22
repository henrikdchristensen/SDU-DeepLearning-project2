import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_scores(results, label):
    _, axes = plt.subplots(1, 3, figsize=(9, 3))  # Adjust the layout for 3 subplots

    epochs = range(1, results["num_epochs"] + 1)

    # Plot loss
    axes[0].plot(epochs, results["train_losses"], marker="o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, results["val_losses"], marker="o", markersize=3, linestyle="--", label="Val Loss")
    axes[0].set_title(f"{label}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")

    # Convert accuracy and F1-score to percentages
    train_accuracies = [acc * 100 for acc in results["train_accuracies"]]
    val_accuracies = [acc * 100 for acc in results["val_accuracies"]]
    train_f1_scores = [f1 * 100 for f1 in results["train_f1_scores"]]
    val_f1_scores = [f1 * 100 for f1 in results["val_f1_scores"]]

    # Plot accuracy
    axes[1].plot(epochs, train_accuracies, marker="o", markersize=3, label="Train Accuracy")
    axes[1].plot(epochs, val_accuracies, marker="o", markersize=3, linestyle="--", label="Val Accuracy")
    axes[1].set_title(f"{label}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(loc="best")

    # Plot F1-score
    axes[2].plot(epochs, train_f1_scores, marker="o", markersize=3, label="Train F1-Score")
    axes[2].plot(epochs, val_f1_scores, marker="o", markersize=3, linestyle="--", label="Val F1-Score")
    axes[2].set_title(f"{label}: F1-Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1-Score (%)")
    axes[2].legend(loc="best")

    # Set the number of ticks on the x and y axes for all plots
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
