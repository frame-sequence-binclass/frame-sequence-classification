import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Linear(self.hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, attn_weights


class CNN_LSTM_Attn(nn.Module):
    def __init__(self, image_size: int, hidden_size: int, num_layers: int, bidirectional: bool, 
                 pretrained: bool = False, dropout: float = 0.3):
        super(CNN_LSTM_Attn, self).__init__()

        if pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.cnn_encoder = resnet
            self.feature_dim = resnet.fc.in_features
            self.cnn_encoder.fc = nn.Identity()
            for param in self.cnn_encoder.parameters():
                param.requires_grad = False
            for param in self.cnn_encoder.layer4.parameters():
                param.requires_grad = True
                
        else:
            # CNN from scratch
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))  # [batch, 512, 1, 1]
            )
            self.feature_dim = 512

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Features Normalization 
        self.lstm_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))

        # Temporal Attention
        self.attention = TemporalAttention(hidden_size, bidirectional)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # (logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, C, H, W = x.shape
        # CNN Encoding
        x_reshaped = x.view(batch_size * sequence_len, C, H, W)
        cnn_features = self.cnn_encoder(x_reshaped)

        # Flatten
        cnn_features = cnn_features.view(batch_size * sequence_len, -1)

        # [batch, seq, feature]
        lstm_input = cnn_features.view(batch_size, sequence_len, self.feature_dim)

        # LSTM
        lstm_output, _ = self.lstm(lstm_input)  # (batch, seq, hidden)
        lstm_output    = self.lstm_norm(lstm_output)

        # Attention
        context, attn_weights = self.attention(lstm_output)

        # Classifier
        logits = self.classifier(context)

        return logits
