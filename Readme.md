# Vietnamese Comment Sentiment Analysis using CNN-BiLSTM-Attention

## 📋 Overview

This project implements a sentiment analysis system for Vietnamese comments using a **hybrid CNN-BiLSTM neural network with Attention mechanism**. The model determines the emotional sentiment (Negative, Neutral, Positive) of Vietnamese text comments, particularly useful for analyzing customer reviews.

**Paper Title:** *Sentiment Analysis of Comments to Determine Author's Emotions Using Hybrid CNN-BiLSTM Model Combined with Attention Mechanism v2*

---

## 🎯 Problem Statement

Sentiment analysis is a critical Natural Language Processing (NLP) task that automatically determines the emotional tone of text data. This project focuses on Vietnamese language sentiment classification, which presents unique challenges due to the language's morphological and syntactic characteristics.

**Key Applications:**
- Customer review analysis for businesses
- Opinion mining from social media
- Brand reputation monitoring
- Service quality assessment

---

## 🏗️ Model Architecture

### Hybrid CNN-BiLSTM-Attention Architecture

The model combines three powerful neural network paradigms:

#### 1. **Embedding Layer**
- **Input:** Variable vocabulary size (built from training data)
- **Embedding Dimension:** 300
- **Dropout:** 0.2
- **Purpose:** Convert text tokens to dense vector representations

#### 2. **Multi-Kernel CNN Layer**
Captures local features and n-gram patterns:
- **3 parallel convolution branches** with kernel sizes: 2, 3, 5
- **128 filters per kernel** (Total output: 384 dimensions)
- **Activation:** ReLU
- **Purpose:** Extract local contextual patterns at different granularities

#### 3. **Bidirectional LSTM (BiLSTM)**
Captures long-range dependencies:
- **Input dimension:** 384 (from CNN concatenation)
- **Hidden size:** 256 per direction
- **Layers:** 2 stacked layers
- **Output dimension:** 512 (256 × 2 directions)
- **Dropout:** 0.3
- **Purpose:** Learn sequential dependencies in both forward and backward directions

#### 4. **Attention Mechanism**
Focuses on important features:
- Computes attention weights over BiLSTM outputs
- Uses softmax normalization
- Produces weighted context vector
- **Purpose:** Highlight the most relevant parts of the sequence

#### 5. **Classification Head**
- **Layer Normalization** on context vector
- **Dropout:** 0.5 (prevents overfitting)
- **Fully Connected:** 512 → 3 output classes
- **Output Classes:** Negative (0), Neutral (1), Positive (2)

### Architecture Diagram
```
Text Input
    ↓
Embedding (300-dim)
    ↓
CNN-3-Kernel (3×128 filters) → 384-dim
    ↓
BiLSTM-2-Layer (256-hidden) → 512-dim
    ↓
Attention Mechanism → Context Vector
    ↓
Layer Norm + Dropout
    ↓
FC Layer (512 → 3)
    ↓
Softmax → Sentiment Class
```

---

## 📊 Dataset & Preprocessing

### Data Format
The dataset is structured with Vietnamese comments/reviews containing:
- **Text field:** Review content in Vietnamese
- **Aspect Ratings:** 10 dimensions across:
  - room, service, location, price, food_and_beverage, amenities, cleanliness, transportation, policy, others

### Sentiment Label Generation
1. Calculate **average aspect score** across all 10 dimensions
2. Map to sentiment classes using threshold-based approach:
   - **Negative:** average_score < -0.2
   - **Neutral:** -0.2 ≤ average_score ≤ 0.2
   - **Positive:** average_score > 0.2

### Preprocessing Pipeline

#### Text Cleaning
1. Convert to lowercase
2. Remove punctuation and special characters
3. Remove extra whitespace
4. **Vietnamese word tokenization** using UndertheSea library (specialized for Vietnamese NLP)

#### Vocabulary Building
- **Minimum frequency threshold:** 2 occurrences
- **Special tokens:**
  - `<PAD>` (ID: 0) - Padding token
  - `<UNK>` (ID: 1) - Unknown word token
- Filter out rare words to reduce vocabulary size

#### Sequence Encoding
- Convert words to token IDs
- Truncate sequences longer than 256 tokens
- Pad shorter sequences with `<PAD>` tokens
- **Max sequence length:** 256

### Class Distribution
The dataset exhibits class imbalance, addressed through weighted loss:
| Class | Weight | Proportion |
|-------|--------|-----------|
| Negative | 3.52 | 7.4% |
| Neutral | 1.28 | 21.0% |
| Positive | 0.52 | 71.6% |

---

## 🚀 Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 32 |
| **Epochs** | 15 (with early stopping) |
| **Sequence Max Length** | 256 |
| **Embedding Dimension** | 300 |
| **LSTM Hidden Size** | 256 |
| **Dropout Rate** | 0.5 |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 1e-5 |
| **Gradient Clipping Max Norm** | 5.0 |
| **Early Stopping Patience** | 4 epochs |

### Optimization Strategy

**Loss Function:**
- CrossEntropyLoss with class weights
- Weights handle imbalanced sentiment distribution

**Optimizer:**
- Adam (lr=0.001, weight_decay=1e-5)
- Gradient clipping (max norm: 5.0) prevents exploding gradients

**Learning Rate Scheduler:**
- ReduceLROnPlateau
- Monitors: Validation F1-score
- Factor: 0.5 (reduce LR by 50%)
- Patience: 2 epochs

### Training Strategy
1. Data split: Train/Validation/Test from separate CSV files
2. Best model selection: Based on **validation F1-score** (macro-averaged)
3. Early stopping: Prevents overfitting when no improvement for 4 consecutive epochs
4. Checkpointing: Saves best model weights during training

---

## 📈 Results & Performance

### Best Model Performance

**Best Epoch:** 7 out of 15

**Validation Metrics (at Best Epoch):**
- Validation F1-Score: **0.8023**
- Validation Accuracy: **0.8551**

**Test Set Results:**
| Metric | Value |
|--------|-------|
| Test Accuracy | **0.8280** |
| Test Precision (macro) | **0.7670** |
| Test Recall (macro) | **0.7819** |
| Test F1-Score (macro) | **0.7736** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | Samples |
|-------|-----------|--------|----------|---------|---------|
| **Negative** | 0.7158 | 0.7312 | 0.7234 | High | 186 |
| **Neutral** | 0.6632 | 0.7362 | 0.6978 | High | 527 |
| **Positive** | 0.9219 | 0.8784 | 0.8996 | Highest | 1316 |

### Key Observations

✅ **Strengths:**
- Excellent performance on Positive sentiment detection (F1: 0.90)
- Good overall accuracy (82.8%)
- Effective handling of class imbalance through weighted loss

⚠️ **Insights:**
- Neutral sentiment is the most challenging class (F1: 0.70)
- Model performs well given significant class imbalance (11.6× more positive samples)
- Negative sentiment detection is moderate but reasonable (F1: 0.72)

### Training Trajectory

| Epoch | Train F1 | Val F1 | Notes |
|-------|----------|--------|-------|
| 1 | 0.6334 | 0.7119 | Initial training |
| 7 | 0.8895 | **0.8023** | **BEST** |
| 10 | 0.9245 | 0.7858 | LR reduced (0.001 → 0.0005) |
| 11 | 0.9318 | 0.7824 | Early stopping triggered |

---

## 📁 Project Structure

```
vietnam_analyze_tw/
├── data/
│   ├── full_data.csv          # Complete dataset
│   ├── train.csv              # Training set
│   ├── val.csv                # Validation set
│   ├── test.csv               # Test set
│   └── test.txt               # Raw test data
│
├── models/
│   ├── __init__.py
│   └── cnn_bilstm_attention.py  # Model architecture
│
├── training/
│   ├── __init__.py
│   ├── train.py                 # Training script
│   └── outputs/
│       ├── checkpoints/
│       │   └── best_model.pt    # Best model weights
│       ├── logs/
│       │   ├── hyperparameters.txt
│       │   ├── best_epoch.txt
│       │   ├── metrics.csv
│       │   ├── training_log.txt
│       │   └── classification_report.txt
│       └── plots/
│           ├── loss_curve.png       # Training/Val loss
│           ├── accuracy_curve.png   # Training/Val accuracy
│           └── confusion_matrix.png # Per-class matrix
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py            # Text preprocessing
│   ├── dataset_loader.py        # Data loading utilities
│   ├── metrics.py               # Metric calculations
│   └── __pycache__/
│
├── predict.py                   # Inference script
├── requirements.txt             # Dependencies
└── Readme.md                    # This file
```

---

## 🔧 Installation & Setup

### Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- UndertheSea (Vietnamese NLP)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd vietnam_analyze_tw

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import underthesea; print('UndertheSea installed successfully')"
```

---

## 🎓 Usage Guide

### 1. Data Preparation

```bash
# The dataset should be in data/ folder:
# - train.csv
# - val.csv
# - test.csv
```

### 2. Training the Model

```bash
cd training
python train.py
```

**Output:**
- Best model saved to `outputs/checkpoints/best_model.pt`
- Training logs and metrics to `outputs/logs/`
- Visualization plots to `outputs/plots/`

### 3. Making Predictions

Run the interactive prediction script:

```bash
python predict.py
```

**Interactive Mode:**
```
Enter Vietnamese text to analyze:
Khách sạn này rất đáng giá. Phòng sạch, nhân viên thân thiện.

Overall Verdict: TÍCH CỰC (Positive)
Confidence: 95% 
Probabilities: Negative: 0.02 | Neutral: 0.03 | Positive: 0.95
Positive Sentences: 2 | Neutral Sentences: 0 | Negative Sentences: 0
```

### 4. Single Text Prediction

```python
from models.cnn_bilstm_attention import Model
import torch

# Load best model
model = Model(vocab_size, embedding_dim=300, hidden_size=256)
model.load_state_dict(torch.load('training/outputs/checkpoints/best_model.pt'))

# Predict
text = "Khách sạn này rất đáng giá"
logits = model(preprocessed_text)
probabilities = torch.softmax(logits, dim=1)
sentiment = torch.argmax(logits)  # 0: Negative, 1: Neutral, 2: Positive
```

---

## 📊 Evaluation Metrics

The model is evaluated on multiple metrics:

### Classification Metrics
- **Accuracy:** Overall correctness rate
- **Precision:** True positive rate among positive predictions
- **Recall:** Proportion of actual positives correctly identified
- **F1-Score:** Harmonic mean of precision and recall

### Averaging Method
- **Macro-averaging:** Equal weight to all classes (important for imbalanced data)
- Prevents bias towards majority class (Positive sentiment)

### Visualization Outputs
1. **Loss curves:** Training vs Validation loss across epochs
2. **Accuracy curves:** Training vs Validation accuracy across epochs
3. **Confusion Matrix:** Per-class prediction patterns (heatmap)
4. **Classification Report:** Detailed per-class statistics

---

## 🔍 Model Insights

### Why This Architecture?

**CNN Component:**
- Detects local patterns and n-gram features efficiently
- Multi-kernel design captures patterns at different scales
- Excellent for capturing idiomatic expressions in Vietnamese

**BiLSTM Component:**
- Captures long-range sequential dependencies
- Bidirectional processing understands context from both directions
- Multiple layers allow hierarchical feature learning

**Attention Mechanism:**
- Identifies which words are most important for sentiment classification
- Provides interpretability by showing attention weights
- Improves performance by focusing on relevant features

**Combined Approach:**
- Synergistic: CNN provides fast local feature extraction, BiLSTM captures dependencies
- Attention provides interpretable focus
- Weighted loss handles class imbalance effectively

### Performance Characteristics

**Strengths:**
- Robust to class imbalance (71.6% positive samples)
- High accuracy overall (82.8%)
- Excellent positive sentiment detection (F1: 0.90)

**Limitations:**
- Neutral sentiment is challenging (F1: 0.70) - semantic boundary is fuzzy
- Smaller training data impacts negative sentiment performance (F1: 0.72)
- Vietnamese-specific preprocessing critical for performance

---

## 🧪 Reproduction & Reproducibility

### Fixed Seed
- Seed value: **42**
- Applied to: NumPy, PyTorch, random, and CUDA
- Ensures deterministic behavior across runs

### Device Configuration
- Automatic GPU detection (CUDA if available)
- Falls back to CPU if CUDA unavailable
- Set `device = 'cuda' if torch.cuda.is_available() else 'cpu'`

### Exact Reproduction
```bash
python training/train.py
# Should produce:
# - best_model.pt with same weights
# - metrics.csv with same values
# - Epoch 7 as best epoch
```

---

## 📈 Future Improvements

1. **Data Augmentation:** Back-translation, synonym replacement for better generalization
2. **Transfer Learning:** Pre-trained Vietnamese embeddings (PhoBERT, ViBERT)
3. **Ensemble Methods:** Combine with other models for improved robustness
4. **Advanced Attention:** Multi-head attention for richer feature combinations
5. **Aspect-based Sentiment:** Extend to 10 aspect-level sentiments instead of combined
6. **Real-time Processing:** Optimize for production deployment
7. **Explainability:** Generate attention visualizations for model interpretation

---

## 📚 References & Dependencies

### Key Libraries
- **PyTorch:** Deep learning framework
- **NumPy/Pandas:** Data manipulation and analysis
- **Scikit-learn:** Metrics and evaluation
- **Matplotlib/Seaborn:** Visualization
- **UndertheSea:** Vietnamese NLP (tokenization, preprocessing)

### Technical Concepts
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM) / Bidirectional LSTM
- Attention Mechanisms
- Sentiment Analysis / Opinion Mining
- Vietnamese Natural Language Processing

---

## 👤 Authors & Contributors

**Project Title:** Sentiment Analysis of Comments to Determine Author's Emotions Using Hybrid CNN-BiLSTM Model Combined with Attention Mechanism v2

**Purpose:** Academic research in Vietnamese sentiment analysis and neural network architecture design

---

## 📝 License

This project is provided for educational and research purposes.

---

## 💬 Questions & Support

For detailed information about the model implementation, training results, and technical specifications, refer to:
- `training/outputs/logs/hyperparameters.txt` - Exact hyperparameters
- `training/outputs/logs/training_log.txt` - Detailed training progress
- `training/outputs/logs/classification_report.txt` - Comprehensive metrics
- `training/outputs/plots/` - Visualization plots

---

**Last Updated:** April 2026 | **Model Version:** CNN-BiLSTM-Attention v2
