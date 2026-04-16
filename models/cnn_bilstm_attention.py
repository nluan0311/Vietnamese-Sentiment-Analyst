import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attention(lstm_outputs), dim=1)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)  # (batch_size, hidden_dim)
        return context_vector, attn_weights


class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, cnn_filters=128, kernel_size=3, lstm_hidden=256, num_classes=3,
                 dropout=0.5):
        super(CNN_BiLSTM_Attention, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)

        # 2. Multi-kernel CNN Layer
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, 2, padding=1),
            nn.Conv1d(embed_dim, 128, 3, padding=1),
            nn.Conv1d(embed_dim, 128, 5, padding=2)
        ])

        # 3. BiLSTM Layer (Capture long sequential context)
        self.bilstm = nn.LSTM(input_size=384, hidden_size=lstm_hidden,
                              num_layers=2, dropout=0.3,batch_first=True, bidirectional=True) #batch_fist=(batch, seq_len, features)

        # 4. Attention Mechanism
        self.attention = AttentionLayer(hidden_dim=lstm_hidden * 2) #256*2

        # 5. Fully Connected & Normalization
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeds = self.embed_dropout(self.embedding(x))

        cnn_in = embeds.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(cnn_in))
            conv_outputs.append(conv_out)

        min_len = min([c.shape[2] for c in conv_outputs])
        conv_outputs = [c[:, :, :min_len] for c in conv_outputs]

        cnn_out = torch.cat(conv_outputs, dim=1)

        # BiLSTM expects (batch, seq_len, features)
        lstm_in = cnn_out.permute(0, 2, 1) # coveter (batch, 384, seq_len) to (batch, seq_len, 384) beacuse input LTSM (batch, seq_len, features)
        lstm_out, _ = self.bilstm(lstm_in)  # (batch, seq_len, lstm_hidden * 2)

        # Apply Attention
        context, attn_weights = self.attention(lstm_out)

        # Normalization, Dropout & Classification
        norm_context = self.layer_norm(context)
        drop_context = self.dropout(norm_context)
        logits = self.fc(drop_context)

        return logits