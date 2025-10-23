import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class CNN3D(nn.Module):
    def __init__(self, sequence_length: int = 5, image_size: int = 224):
        super(CNN3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        self._get_flattened_size(sequence_length, image_size)
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1),
        )

    def _get_flattened_size(self, sequence_length: int, image_size: int):
        with torch.no_grad():
            # The input order is batch, channels, sequence, height, width
            dummy_input = torch.randn(1, 3, sequence_length, image_size, image_size)
            dummy_output = self.conv_block(dummy_input)
            self.flattened_size = torch.flatten(dummy_output, 1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

class Binary3DCNN(nn.Module):
    def __init__(self, model_type, sequence_length, img_size, num_classes=1):
        super().__init__()
        self.model_type = model_type

        if model_type == 'scratch':
            self.base_model = CNN3D(sequence_length=sequence_length, image_size=img_size)
        elif model_type == 'pretrained':
            self.base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
            # Freeze pretrained weights
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True
            
            # Replace the final fully connected layer
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            raise ValueError(f"Model type '{model_type}' not supported.")

    def forward(self, x):
        if self.model_type == 'scratch':
            return self.base_model(x)
        else:
            return self.base_model(x)