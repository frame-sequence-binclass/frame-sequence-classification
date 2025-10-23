import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class HybridCNNTransformer(nn.Module):
    def __init__(self, sequence_length: int = 10, image_size: int = 224,
                 embed_dim: int = 512, num_heads: int = 8, mlp_dim: int = 2048,
                 num_layers: int = 4, pretrained: bool = True, dropout: float = 0.3):
        super(HybridCNNTransformer, self).__init__()

        # CNN Feature Extractor
        if pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.cnn_extractor = resnet
            self.feature_dim = resnet.fc.in_features
            self.cnn_extractor.fc = nn.Identity()
            for param in self.cnn_extractor.parameters():
                param.requires_grad = False
            for param in self.cnn_extractor.layer4.parameters():
                param.requires_grad = False
        else:
            # CNN from scratch
            self.cnn_extractor = nn.Sequential(
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
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 512

        self.cnn_output_size = self.feature_dim

        # Embedding projection
        self.projection = nn.Linear(self.cnn_output_size, embed_dim)

        # Tokens & Positional Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length + 1, embed_dim))

        # Transformer Encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, C, H, W = x.shape

        x_reshaped = x.view(batch_size * sequence_len, C, H, W)

        # [batch*seq, 512, 1, 1]
        cnn_features = self.cnn_extractor(x_reshaped)
        cnn_features = cnn_features.view(batch_size * sequence_len, -1)  # [batch*seq, 512]

        # Projection -> embedding
        projected_features = self.projection(cnn_features)

        # [batch, seq, embed_dim]
        transformer_input = projected_features.view(batch_size, sequence_len, -1)

        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((cls_tokens, transformer_input), dim=1)

        # Positional embeddings
        transformer_input = transformer_input + self.pos_embedding[:, :sequence_len + 1, :]

        # Transformer
        transformer_output = self.transformer_encoder(transformer_input)

        cls_output = transformer_output[:, 0, :]

        # Classifier
        logits = self.classifier(cls_output)

        return logits
