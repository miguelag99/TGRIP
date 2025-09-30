import torch
import torch.nn.functional as F

from einops import rearrange

from tgrip.loss import LossInterface


class CosineSimilarityLoss(LossInterface):
    def __init__(
        self,
        key="semantic_bev",
        name="loss_semantic_bev",
    ):
        """
        Cosine Similarity Loss between predicted and ground truth BEV semantic features.
        """
        super().__init__(key=key, name=name)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            pred (torch.Tensor): BEV semantic features of shape [B, C, H, W].
            gt (torch.Tensor): BEV semantic gt features of shape [B, C, H, W].
        """
                
        preds = rearrange(preds, 'b c h w -> b (h w) c')
        targets = rearrange(targets, 'b c h w -> b (h w) c')

        mask = (targets.abs().sum(dim=-1) != 0)  # [B, H*W]

        # Normalize embeddings along channel dimension
        preds = F.normalize(preds[mask], dim=1)
        targets = F.normalize(targets[mask], dim=1)

        loss = 1 - F.cosine_similarity(preds, targets, dim=1).mean()
    
        return loss