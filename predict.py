import torch
import torch.nn.functional as F
import os
import sys
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.cnn_bilstm_attention import CNN_BiLSTM_Attention
from utils.preprocess import clean_vietnamese_text, text_to_sequence
from utils.dataset_loader import VietnameseSentimentDataset

print("Dang khoi dong mo hinh...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
EMBED_DIM = 300
HIDDEN_SIZE = 256
DROPOUT = 0.5

train_path = os.path.join(BASE_DIR, 'data', 'train.csv')
dataset = VietnameseSentimentDataset(train_path, is_train=True)
vocab = dataset.vocab

model_path = os.path.join(BASE_DIR, 'training', 'outputs', 'checkpoints', 'best_model.pt')
model = CNN_BiLSTM_Attention(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    lstm_hidden=HIDDEN_SIZE,
    dropout=DROPOUT
).to(DEVICE)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Da nap model thanh cong.")
else:
    print("Khong tim thay best_model.pt")
    sys.exit()

label_map = {
    0: "TIEU CUC",
    1: "TRUNG TINH",
    2: "TICH CUC"
}


def predict_one_text(text: str):
    cleaned_text = clean_vietnamese_text(text)
    seq = text_to_sequence(cleaned_text, vocab, max_len=MAX_LEN)
    input_tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_class = torch.argmax(probs).item()

    return {
        "label_id": pred_class,
        "label_name": label_map[pred_class],
        "prob_neg": probs[0].item() * 100,
        "prob_neu": probs[1].item() * 100,
        "prob_pos": probs[2].item() * 100,
        "confidence": probs[pred_class].item() * 100
    }


def split_sentences(text: str):
    raw_sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 5]


def predict_sentiment(text: str):
    result = predict_one_text(text)
    sentences = split_sentences(text)

    so_cau_khen = 0
    so_cau_che = 0
    so_cau_trung_tinh = 0

    for s in sentences:
        s_result = predict_one_text(s)
        if s_result["label_id"] == 2:
            so_cau_khen += 1
        elif s_result["label_id"] == 0:
            so_cau_che += 1
        else:
            so_cau_trung_tinh += 1

    print("\n" + "=" * 70)
    print("KET QUA PHAN TICH CAM XUC")
    print("=" * 70)
    print(f"Van ban goc         : {text.strip()}")
    print("-" * 70)
    print(f"Ket luan tong the   : {result['label_name']}")
    print(f"Do tu tin mo hinh   : {result['confidence']:.2f}%")
    print("-" * 70)
    print(f"Ti le Tich cuc      : {result['prob_pos']:.2f}%")
    print(f"Ti le Tieu cuc      : {result['prob_neg']:.2f}%")
    print(f"Ti le Trung tinh    : {result['prob_neu']:.2f}%")
    print("-" * 70)
    print("THONG KE THAM KHAO THEO TUNG CAU:")
    print(f"So cau Khen         : {so_cau_khen}")
    print(f"So cau Che          : {so_cau_che}")
    print(f"So cau Trung tinh   : {so_cau_trung_tinh}")
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "*" * 70)
    print("CHUONG TRINH PHAN TICH CAM XUC VAN BAN")
    print("Nhap doan review, nhan Enter 2 lan de phan tich.")
    print("Go 'exit' de thoat.")
    print("*" * 70)

    while True:
        print("\nMoi ban nhap doan van:")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == 'exit' and len(lines) == 0:
                print("Da thoat chuong trinh.")
                sys.exit(0)
            if line == "":
                break
            lines.append(line)

        doan_van_test = " ".join(lines)
        if len(doan_van_test.strip()) < 2:
            continue

        predict_sentiment(doan_van_test)