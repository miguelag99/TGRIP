import torch
import torch.nn as nn

from einops import rearrange

from tgrip.utils.debug import debug_hook
    
class CLIPSemanticFuser(nn.Module):
    """
    Module focused on fusing BEV features with text embeddings using cross-attention.
    First, a semantic projector maps from BEV feature space to text embedding space.
    Then, cross-attention is applied to fuse the two modalities and filter the BEV features
    """
    def __init__(
        self,
        embed_dim=512,
        bev_dim=128,
        num_heads=8,
        mlp_dim=256,
        past_frames=3,
        projector_layers=1,
    ):
        super().__init__()
        self.semantic_projector = SemanticProjector(
            bev_dim=bev_dim,
            past_frames=past_frames,
            text_dim=embed_dim,
            hidden_dim=mlp_dim,
            n_hidden_layers=projector_layers,
            kernel_size=1,
            stride=1,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.final_fuser = nn.Conv2d(
            in_channels=bev_dim+1,
            out_channels=bev_dim,
            kernel_size=1,
            stride=1
        )


    def forward(self, bev_feats, text_embed):
        """
        BEV past features: [B, T, C, H, W]
        text_embed: CLIP embedding [B, N_text, D]
        
        If multiple text embeddings are provided, attention weights are averaged.
        
        Returns:
        semantic_bev: [B, D, H, W] — BEV features projected to text embedding space
        final_feats: [B, T, C, H, W] — BEV features fused with text attention scores
        """
        B, T, C, H, W = bev_feats.shape
        semantic_bev = self.semantic_projector(bev_feats)  # [B, D, H, W]
        
        if text_embed is not None:
            bev_flat = semantic_bev.flatten(2).permute(0, 2, 1)  # [B, N, D], N = H*W
            text_q, attn_weights = self.cross_attn(
                query=text_embed, key=bev_flat, value=bev_flat
            )
            
            # Fuse attention weights for different text tokens by averaging
            text_score = attn_weights.mean(1, keepdims=True).view(B, 1, H, W)  # [B, H, W]
            text_score = text_score.unsqueeze(1)  # [B, 1, 1, H, W]
            text_score = text_score.expand(-1, T, 1, -1, -1)  # [B, T, 1, H, W]
        else:
            text_score = torch.ones(B, T, 1, H, W, device=bev_feats.device)
            # If no text condition is provided, assume uniform attention over the BEV space
        
        final_feats = self.final_fuser(
            torch.cat([bev_feats, text_score], dim=2).view(B*T, C+1, H, W)
        )
        final_feats = final_feats.view(B, T, -1, H, W)
        
        return semantic_bev, final_feats
    
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