import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFiLMModulation(nn.Module):
    def __init__(self, text_dim=512, bev_dim=128, num_layers=3, use_norm=False):
        """
        text_dim: Dimensionality of the text embedding (e.g., 512 from CLIP)
        bev_dim: Number of BEV feature channels
        num_layers: Number of FiLM layers to apply
        use_norm: Whether to use normalization (e.g., LayerNorm) in each FiLM layer
        """
        super(MultiFiLMModulation, self).__init__()
        self.num_layers = num_layers
        self.use_norm = use_norm

        # Each FiLM layer: text → (gamma, beta) projection
        self.film_layers = nn.ModuleList([
            nn.Linear(text_dim, bev_dim * 2) for _ in range(num_layers)
        ])

        if use_norm:
            self.norms = nn.ModuleList([
                nn.LayerNorm([bev_dim, 1, 1]) for _ in range(num_layers)
            ])

    def forward(self, bev_feats, text_feat):
        """
        bev_feats: [B, C, H, W]
        text_feat: [B, text_dim]
        """
        x = bev_feats
        for i in range(self.num_layers):
            gamma_beta = self.film_layers[i](text_feat)  # [B, 2*C]
            gamma, beta = gamma_beta.chunk(2, dim=-1)    # [B, C] each

            # Reshape for broadcast
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)      # [B, C, 1, 1]

            if self.use_norm:
                x = self.norms[i](x)

            x = gamma * x + beta
            x = F.relu(x)  # Optional: non-linearity between layers

        return x