import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.preprocess import clean_vietnamese_text, text_to_sequence


class VietnameseSentimentDataset(Dataset):
    def __init__(self, csv_path, vocab=None, max_len=200, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.is_train = is_train

        # Tiền xử lý văn bản
        print(f"Preprocessing {csv_path}...")
        self.df['processed_text'] = self.df['data'].apply(clean_vietnamese_text)

        if is_train and vocab is None:
            from utils.preprocess import build_vocab
            self.vocab = build_vocab(self.df['processed_text'].tolist())
        else:
            self.vocab = vocab

        # Chuyển đổi điểm ordinal thành 3 nhãn: 0 (Negative), 1 (Neutral), 2 (Positive)
        aspect_cols = ['room', 'service', 'location', 'price', 'food_and_beverage',
                       'amenities', 'cleanliness', 'transportation', 'policy', 'others']

        self.df[aspect_cols] = self.df[aspect_cols].apply(pd.to_numeric, errors='coerce')
        # Tính trung bình các aspect
        self.df['avg_score'] = self.df[aspect_cols].mean(axis=1)


        def map_sentiment(score):
            if score < -0.2:
                return 0  # Negative
            elif score > 0.2:
                return 2  # Positive
            else:
                return 1  # Neutral

        self.labels = self.df['avg_score'].apply(map_sentiment).tolist()
        self.texts = self.df['processed_text'].tolist()

        # Kiểm tra phân bố nhãn
        print("Label distribution:")
        print(pd.Series(self.labels).value_counts())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = text_to_sequence(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def get_dataloaders(train_path, val_path, test_path, batch_size=32, max_len=200):
    train_dataset = VietnameseSentimentDataset(train_path, max_len=max_len, is_train=True)
    vocab = train_dataset.vocab

    val_dataset = VietnameseSentimentDataset(val_path, vocab=vocab, max_len=max_len, is_train=False)
    test_dataset = VietnameseSentimentDataset(test_path, vocab=vocab, max_len=max_len, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab