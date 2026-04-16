import re
import string
from underthesea import word_tokenize


def clean_vietnamese_text(text):
    """Làm sạch và tách từ tiếng Việt"""
    if not isinstance(text, str):
        return ""
    # Chuyển chữ thường
    text = text.lower()
    # Xóa dấu câu và ký tự đặc biệt
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    # Tách từ tiếng Việt bằng underthesea
    text = word_tokenize(text, format="text")
    return text


def build_vocab(texts, min_freq=2):
    """Xây dựng từ điển (Vocabulary) từ tập dữ liệu"""
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text, vocab, max_len=200):
    """Chuyển text thành chuỗi số nguyên có độ dài cố định"""
    tokens = text.split()
    seq = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    # Truncate hoặc Pad
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
    return seq