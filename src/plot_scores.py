import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_scores(results, label):
    # Create subplots for loss and accuracy
    _, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjusted size for better clarity

    epochs = range(1, results["num_epochs"] + 1)

    # Plot loss
    axes[0].plot(epochs, results["train_losses"], marker="o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, results["val_losses"], marker="o", markersize=3, linestyle="--", label="Val Loss")
    axes[0].set_title(f"{label}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")

    # Plot accuracy
    axes[1].plot(epochs, results["train_accuracies"], marker="o", markersize=3, label="Train Accuracy")
    axes[1].plot(epochs, results["val_accuracies"], marker="o", markersize=3, linestyle="--", label="Val Accuracy")
    axes[1].set_title(f"{label}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(loc="best")

    # Set the number of ticks on the x and y axes
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
