"""Hybrid CNN-Vision Transformer for radiomic feature extraction."""
import torch
import torch.nn as nn


class CNNVisionTransformer(nn.Module):
    """CNN backbone followed by Vision Transformer for radiomic feature extraction."""

    def __init__(
        self,
        input_channels: int = 1,
        patch_size: int = 16,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        dropout: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = image_size

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.patch_embed = nn.Conv2d(
            256, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        # CNN backbone reduces spatial dims by 4x (stride 2 + maxpool 2)
        reduced = image_size // 4
        num_patches = (reduced // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn_backbone(x)
        patches = self.patch_embed(cnn_features)
        patches = patches.flatten(2).transpose(1, 2)

        batch_size = patches.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        patches = patches + self.pos_embed[:, : patches.shape[1], :]

        transformer_out = self.transformer(patches)
        cls_repr = transformer_out[:, 0, :]
        return self.output_proj(cls_repr)