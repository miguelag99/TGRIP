import torch
import torch.nn.functional as F

from torchmetrics.metric import Metric
from einops import rearrange


class CosineSimilarityMetric(Metric):
    """
    Computes the average cosine similarity between pred and target in BEV maps [B, C, H, W].
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_cos", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target: [B, C, H, W]
        assert preds.shape == target.shape, "preds and target should have the same shape"
        
        preds = rearrange(preds, 'b c h w -> b (h w) c')
        target = rearrange(target, 'b c h w -> b (h w) c')

        mask = (target.abs().sum(dim=-1) != 0)  # [B, H*W]
        
        # Normalize embeddings along channel dimension
        preds = F.normalize(preds[mask], dim=1)
        target = F.normalize(target[mask], dim=1)

        # Compute cosine similarity for each cell → [B, H, W]
        cos = F.cosine_similarity(preds, target, dim=1)

        # Accumulate sum of similarities and total count
        self.sum_cos += cos.sum()
        self.total += cos.numel()

    def compute(self):
        return self.sum_cos / self.total
