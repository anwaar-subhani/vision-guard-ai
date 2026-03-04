"""
LSTM-based model for posture sequence classification.
Detects: normal (standing), fall, lying/crawling
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PostureLSTM(nn.Module):
    """
    LSTM model for classifying posture sequences.
    Input: (batch, sequence_length, features) - pose keypoints over time
    Output: (batch, num_classes) - probabilities for each posture class
    """
    def __init__(self, input_size=66, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        """
        Args:
            input_size: Number of features per timestep (33 landmarks * 2 coords = 66)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (normal, fall, lying)
            dropout: Dropout rate
        """
        super(PostureLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM to capture temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism to focus on important timesteps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size) - pose sequences
        Returns:
            logits: (batch, num_classes) - classification logits
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classification
        out = F.relu(self.fc1(attended))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
    
    def predict_proba(self, x):
        """Returns class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

class PostureClassifier:
    """Wrapper class for easier inference."""
    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PostureLSTM()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = ["normal", "fall", "lying"]
        
    def predict(self, sequence):
        """
        Predict posture class for a sequence.
        Args:
            sequence: numpy array of shape (seq_len, 66) or torch tensor
        Returns:
            class_name: str, predicted class
            confidence: float, confidence score
        """
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence).float()
        
        # Add batch dimension if needed
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        
        sequence = sequence.to(self.device)
        
        with torch.no_grad():
            probs = self.model.predict_proba(sequence)
            confidence, pred_idx = torch.max(probs, dim=1)
        
        class_name = self.class_names[pred_idx.item()]
        confidence_score = confidence.item()
        
        return class_name, confidence_score

