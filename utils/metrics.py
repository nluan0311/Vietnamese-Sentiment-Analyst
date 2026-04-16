import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_metrics(true_labels, preds):
    """
    Tính các metric chính cho bài toán phân loại 3 lớp.
    Sử dụng macro average để phù hợp với dữ liệu mất cân bằng.
    """
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        preds,
        average='macro',
        zero_division=0
    )
    return acc, precision, recall, f1


def save_training_plots(train_losses, val_losses, train_accs, val_accs, save_dir="outputs/plots"):
    """
    Lưu biểu đồ Loss và Accuracy theo epoch.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(true_labels, preds, classes=None, save_dir="outputs/plots"):
    """
    Lưu confusion matrix với thứ tự lớp cố định:
    0 = Negative, 1 = Neutral, 2 = Positive
    """
    if classes is None:
        classes = ["Negative", "Neutral", "Positive"]

    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()