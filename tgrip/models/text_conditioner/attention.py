import torch
import torch.nn as nn
import torch.nn.functional as F

from tgrip.utils.debug import debug_hook

class TextCrossAttention(nn.Module):
    def __init__(self, bev_dim=128, text_dim=512, patch_size=10, attn_heads=4):
        """
        bev_dim: BEV feature channels
        text_dim: CLIP embedding size
        patch_size: patch size (e.g., 10 → 20×20 tokens for 200×200 input)
        attn_heads: number of attention heads
        """
        super().__init__()
        self.downsample = nn.Conv2d(
            bev_dim, bev_dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_proj = nn.Linear(text_dim, bev_dim)
        self.cross_attn = nn.MultiheadAttention(bev_dim, attn_heads, batch_first=True)
        self.patch_size = patch_size
        
        self.register_forward_hook(debug_hook)

    def forward(self, bev_feats, text_feat):
        """
        bev_feats: [B, C, H, W]
        text_feat: [B, text_dim]
        returns: [B, C, H, W] — modulated BEV features
        """
        B, C, H, W = bev_feats.shape
        H_ds, W_ds = H // self.patch_size, W // self.patch_size

        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), "BEV dimensions must be divisible by patch size"

        # Downsample to patch tokens
        bev_patches = self.downsample(bev_feats)  # [B, C, H/patch, W/patch]
        bev_tokens = bev_patches.flatten(2).transpose(1, 2)
        
        text_tokens = self.text_proj(text_feat).unsqueeze(1)  # [B, 1, C]
        text_tokens = text_tokens.expand(B, bev_tokens.shape[1], C)
        
        # Cross-attention: BEV tokens attend to text
        attn_out, attn_weights = self.cross_attn(
            query=bev_tokens, key=text_tokens, value=text_tokens, need_weights=True
        )  # [B, N, C]
        fused_tokens = bev_tokens + attn_out  # residual connection

        # Reshape back to [B, C, H_ds, W_ds]
        fused = fused_tokens.transpose(1, 2).reshape(B, C, H_ds, W_ds)

        # Upsample to original BEV size
        fused_upsampled = F.interpolate(
            fused, size=(H, W), mode="bilinear", align_corners=False
        )

        return fused_upsampled, attn_weights