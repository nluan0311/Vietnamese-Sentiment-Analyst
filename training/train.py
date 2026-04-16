import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

# Chỉnh path cho PyCharm để có thể import từ project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import get_dataloaders
from models.cnn_bilstm_attention import CNN_BiLSTM_Attention
from utils.metrics import compute_metrics, save_training_plots, save_confusion_matrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_hyperparameters(path: str, config: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== CẤU HÌNH HUẤN LUYỆN ===\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")


def train_model():
    # =========================
    # 1. Hyperparameters
    # =========================
    SEED = 42
    BATCH_SIZE = 32
    MAX_LEN = 256
    EMBED_DIM = 300
    HIDDEN_SIZE = 256
    EPOCHS = 15
    LR = 0.001
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 4
    GRAD_CLIP = 5.0
    DROPOUT = 0.5
    CLASS_WEIGHTS = [3.52, 1.28, 0.52]  # [Negative, Neutral, Positive]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)
    print(f"Sử dụng thiết bị: {DEVICE}")

    # =========================
    # 2. Tạo thư mục output
    # =========================
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    config = {
        "SEED": SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_LEN": MAX_LEN,
        "EMBED_DIM": EMBED_DIM,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "GRAD_CLIP": GRAD_CLIP,
        "DROPOUT": DROPOUT,
        "CLASS_WEIGHTS": CLASS_WEIGHTS,
        "BEST_MODEL_METRIC": "val_f1"
    }
    save_hyperparameters("outputs/logs/hyperparameters.txt", config)

    # =========================
    # 3. Load dữ liệu
    # =========================
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        r"E:\225\MC\vietnam_analyze_tw\data\train.csv",
        r"E:\225\MC\vietnam_analyze_tw\data\val.csv",
        r"E:\225\MC\vietnam_analyze_tw\data\test.csv",
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )
    vocab_size = len(vocab)
    print(f"Kích thước vocab: {vocab_size}")

    # =========================
    # 4. Model / Loss / Optimizer
    # =========================
    model = CNN_BiLSTM_Attention(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        lstm_hidden=HIDDEN_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)

    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Scheduler nhẹ để ổn định val
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # theo val_f1
        factor=0.5,
        patience=2
    )

    # =========================
    # 5. Tracking
    # =========================
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    metrics_history = []

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    log_file = open("outputs/logs/training_log.txt", "w", encoding="utf-8")

    # =========================
    # 6. Train loop
    # =========================
    for epoch in range(EPOCHS):
        # ----- TRAIN -----
        model.train()
        running_train_loss = 0.0
        train_preds, train_labels = [], []

        for texts, labels in train_loader:
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        train_loss = running_train_loss / len(train_loader)
        train_acc, train_prec, train_rec, train_f1 = compute_metrics(train_labels, train_preds)

        # ----- VALIDATION -----
        model.eval()
        running_val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(texts)
                loss = criterion(logits, labels)

                running_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_labels, val_preds)

        scheduler.step(val_f1)

        # Lưu lịch sử
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "train_prec": train_prec,
            "train_rec": train_rec,
            "train_f1": train_f1,
            "val_acc": val_acc,
            "val_prec": val_prec,
            "val_rec": val_rec,
            "val_f1": val_f1,
            "lr": current_lr
        })

        log_msg = (
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}\n"
        )
        print(log_msg.strip())
        log_file.write(log_msg)
        log_file.flush()

        # ----- Save best model theo VAL_F1 -----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")
            save_confusion_matrix(val_labels, val_preds)

            print(f"-> Đã lưu best model tại epoch {best_epoch} với Val F1 = {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"-> Không cải thiện. Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # ----- Early stopping -----
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping được kích hoạt.")
            break

    log_file.close()

    # =========================
    # 7. Lưu metrics + plots
    # =========================
    pd.DataFrame(metrics_history).to_csv("outputs/logs/metrics.csv", index=False)
    save_training_plots(train_losses, val_losses, train_accs, val_accs)

    with open("outputs/logs/best_epoch.txt", "w", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation F1: {best_val_f1:.4f}\n")

    # =========================
    # 8. Đánh giá trên test
    # =========================
    print("\nĐang đánh giá mô hình tốt nhất trên tập Test...")

    best_model = CNN_BiLSTM_Attention(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        lstm_hidden=HIDDEN_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)

    best_model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt", map_location=DEVICE))
    best_model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = best_model(texts)
            preds = torch.argmax(logits, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc, test_prec, test_rec, test_f1 = compute_metrics(test_labels, test_preds)
    target_names = ["Negative", "Neutral", "Positive"]
    report = classification_report(test_labels, test_preds, target_names=target_names, digits=4)

    print("\nKẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1-score : {test_f1:.4f}")
    print(report)

    with open("outputs/logs/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best val_f1: {best_val_f1:.4f}\n\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall   : {test_rec:.4f}\n")
        f.write(f"Test F1-score : {test_f1:.4f}\n\n")
        f.write(report)

    print("\nHuấn luyện hoàn tất! Các file đã được lưu vào thư mục 'outputs/'.")


if __name__ == "__main__":
    train_model()