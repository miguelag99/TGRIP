import torch
import torch.nn as nn

from einops import rearrange

from tgrip.utils.debug import debug_hook
    
class CASemanticFuser(nn.Module):
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
        future_frames=6,
        aux_seg_classes_w=[1.0, 2.0],
        projector_layers=1,
    ):
        super().__init__()

        self.class_weights = aux_seg_classes_w
        self.future_frames = future_frames

        self.semantic_projector = SemanticProjector(
            bev_dim=bev_dim,
            past_frames=past_frames,
            future_frames=future_frames,
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
        
        self.aux_seg_head = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                bev_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(bev_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bev_dim,
                len(self.class_weights),
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, bev_feats, text_embed):
        """
        BEV past features: [B, T, C, H, W]
        text_embed: text embedding [B, 1, D]
        
        If multiple text embeddings are provided, attention weights are averaged.
        
        Returns:
        semantic_bev: [B, D, H, W] — BEV features projected to text embedding space
        final_feats: [B, T_out, C, H, W] — BEV features fused with text attention scores
        """
        B, T, C, H, W = bev_feats.shape
        semantic_bev = self.semantic_projector(bev_feats)  # [B, T_out, D, H, W]
        if text_embed is not None:
            text_embed = text_embed.repeat(1, self.future_frames, 1).unsqueeze(2)  # [B, T_out, 1, D]
            bev_flat = semantic_bev.flatten(3).permute(0, 1, 3, 2)  # [B, T_out, N, D]
            bev_q, _ = self.cross_attn(
                query=rearrange(bev_flat, "b to n d -> (b to) n d"),  # [B*T_out, N, D]
                key=rearrange(text_embed, "b to 1 d -> (b to) 1 d"),  # [B*T_out, 1, D]
                value=rearrange(
                    text_embed, "b to 1 d -> (b to) 1 d"
                ),  # [B*T_out, 1, D]
            )
            bev_q = rearrange(
                bev_q, "(b to) (h w) d -> (b to) d h w", to=self.future_frames, h=H, w=W
            )  # [B*T_out, D, H, W]
        else:
            bev_q = rearrange(
                semantic_bev,
                "b to d h w -> (b to) d h w",
            )
            # If no text condition is provided, assume no attention

        text_conditioned_seg = self.aux_seg_head(bev_q)
        text_conditioned_seg = rearrange(
            text_conditioned_seg,
            "(b to) c h w -> b to c h w",
            b=B,
            to=self.future_frames,
        )

        return semantic_bev, text_conditioned_seg
    
class SemanticProjector(nn.Module):
    def __init__(
        self,
        bev_dim: int = 128,
        past_frames: int = 3,
        future_frames: int = 6,
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
                text_dim*future_frames,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(text_dim*future_frames),
            nn.ReLU(inplace=True)
        )
        self.future_frames = future_frames



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

        return rearrange(x, 'b (t d) h w -> b t d h w', t=self.future_frames)


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