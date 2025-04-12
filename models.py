import torch
import torch.nn as nn
from transformers import BertModel

# CNN for Genome Sequences
class CNNGenome(nn.Module):
    def __init__(self, sequence_length):
        super(CNNGenome, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        conv_output_size = (sequence_length - 2) // 2  # Adjust for Conv+Pool
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * conv_output_size, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# BiLSTM for Expression Data
class BiLSTMExpression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMExpression, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output[:, -1, :]  # Last hidden state

# GCN for Protein Networks
class GCNProtein(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNProtein, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        """
        x: feature matrix of shape (num_nodes, feature_dim)
        adj: adjacency matrix of shape (num_nodes, num_nodes)
        """
        x = self.relu(self.gc1(adj @ x))  # Apply graph convolution
        x = self.relu(self.gc2(adj @ x))  # Apply second graph convolution
        return x

# Clinical Data BERT Encoder
class BERTClinical(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased'):
        super(BERTClinical, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output  # [CLS] token representation

# Fusion Network
class MpoxNetFusion(nn.Module):
    def __init__(self, dims):
        super(MpoxNetFusion, self).__init__()
        input_dim = sum(dims)
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()

        # Separate heads
        self.detect_head = nn.Linear(512, 1)
        self.stage_head = nn.Linear(512, 3)
        self.treat_head = nn.Linear(512, 3)

    def forward(self, g_feat, e_feat, p_feat, c_feat):
        x = torch.cat([g_feat, e_feat, p_feat, c_feat], dim=1)
        x = self.relu(self.fc1(x))
        detect = self.detect_head(x)
        stage = self.stage_head(x)
        treat = self.treat_head(x)
        return detect, stage, treat
