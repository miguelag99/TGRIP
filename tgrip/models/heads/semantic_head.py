import torch
import torch.nn as nn

from einops import rearrange

from tgrip.utils.debug import debug_hook


class SemanticHead(nn.Module):
    """
    SemanticHead aligns BEV features with CLIP text embeddings using residual blocks.

    Args:
        bev_dim (int): Number of BEV feature channels.
        past_frames (int): Number of past frames to concatenate.
        text_dim (int): Output embedding dimension (e.g., CLIP embedding size).
        hidden_dim (int): Hidden layer dimension.
        n_hidden_layers (int): Number of residual hidden layers.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(
        self,
        bev_dim: int = 128,
        past_frames: int = 3,
        text_dim: int = 512,
        hidden_dim: int = 256,
        n_hidden_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = bev_dim * past_frames

        self.in_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layers = nn.ModuleList([
            SemanticResBlock(hidden_dim, dropout=dropout) for _ in range(n_hidden_layers)
        ])

        self.out_layer = nn.Conv2d(hidden_dim, text_dim, kernel_size=1)

        self.register_forward_hook(debug_hook)

    def forward(self, bev_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SemanticHead.

        Args:
            bev_feats (torch.Tensor): Input BEV features of shape [B, T, C, H, W].

        Returns:
            torch.Tensor: Output present aligned features of shape [B, 1, text_dim, H, W].
        """
        x = self.in_layer(rearrange(bev_feats, 'b t c h w -> b (t c) h w'))
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Additional residual connection

        x = self.out_layer(x)
        x = x.unsqueeze(1)  # Add time dimension back: [B, 1, text_dim, H, W]
        
        return x
    
    
class SemanticResBlock(nn.Module):
    """
    Residual block for 2D feature maps with spatial context preservation.

    Ensures input and output spatial dimensions are equal (H_in == H_out, W_in == W_out).
    Uses two 3x3 convolutions with padding=1, batch normalization, ReLU activations, and dropout.

    Args:
        channels (int): Number of input/output channels.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        out = self.block(x)
        out = out + x  # Residual connection
        out = self.final_relu(out)
        return out