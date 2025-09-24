import torch.nn as nn

from einops import rearrange

from tgrip.utils.debug import debug_hook
    
class SemanticProjector(nn.Module):
    def __init__(
        self,
        bev_dim: int = 128,
        past_frames: int = 3,
        text_dim: int = 512,
        hidden_dim: int = 256,
        n_hidden_layers: int = 1,
        kernel_size : int = 1,
        stride: int = 1,
    ):
        """
        bev_dim: BEV feature channels
        past_frames: number of past frames
        text_dim: CLIP embedding size
        n_hidden_layers: number of hidden layers
        kernel_size: conv kernel size
        stride: conv stride
        """
        super().__init__()

        self.n_hidden_layers = n_hidden_layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.in_layer = nn.Sequential(
            nn.Conv2d(
                bev_dim*past_frames,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        
        # Hidden layers
        for _ in range(n_hidden_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        stride=stride
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )
            
        # Final layer
        self.out_layer =  nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                text_dim,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(text_dim),
            nn.ReLU(inplace=True)
        )



        self.register_forward_hook(debug_hook)

    def forward(self, bev_feats):
        """
        bev_feats: [B, T, C, H, W]
        returns: [B, text_dim, H, W] — alineated BEV features
        """
        B, T, C, H, W = bev_feats.shape
        
        x = self.in_layer(rearrange(bev_feats, 'b t c h w -> b (t c) h w'))
        for layer in self.layers:
            pre_x = x
            x = layer(x)
            x = x + pre_x  # Residual connection

        x = self.out_layer(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """Squeeze-and-Excitation Layer.
        
        Args:
            channel (int): Number of input channels
            reduction (int): Reduction ratio for the squeeze operation
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze operation
        y = self.avg_pool(x).view(b, c)
        # Excitation operation
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)