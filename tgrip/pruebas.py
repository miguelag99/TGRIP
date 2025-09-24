import hydra
import matplotlib
import matplotlib.pyplot as plt
import pyrootutils
import pytorch_lightning as L
import torch
import torch.nn as nn
from torchmetrics.metric import Metric


from einops import rearrange
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from typing import  Dict, Optional

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tgrip import utils
from tgrip.visualize_batch import generate_gt_instance_pred


log = utils.get_pylogger(__name__)

import torch.nn.functional as F

def cos_similarity_loss(pred: torch.Tensor, gt: torch.Tensor):
    """Compute the cosine similarity loss between predicted and ground truth features.

    Args:
        pred (torch.Tensor): _description_
        gt (Dict[str, torch.Tensor]): Dict containing the different ground truth
            semantic features.
    """
    
    pred = F.normalize(pred, dim=1)
    gt = F.normalize(gt, dim=1)

    return 1 - F.cosine_similarity(pred, gt, dim=1).mean()

class CosineSimilarity(Metric):
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
        
        # normalizar embeddings
        preds = F.normalize(preds, dim=1)
        target = F.normalize(target, dim=1)

        # similitud coseno celda a celda → [B, H, W]
        cos = F.cosine_similarity(preds, target, dim=1)

        # acumular suma y contador
        self.sum_cos += cos.sum()
        self.total += cos.numel()

    def compute(self):
        return self.sum_cos / self.total

def pruebas(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg.data.prefetch_factor = None
    cfg.data.num_workers = 0
    cfg.data.batch_size = 2
    cfg.data.normalize_img = False
    cfg.data.version = 'mini'

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataset = datamodule.train_dataloader()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)

    model = hydra.utils.instantiate(cfg.model).to(device)
    model = utils.load_state_model(
        model,
        ckpt,
        cfg.ckpt.model.freeze,
        cfg.ckpt.model.load,
        verbose=1,
    ).to(device)

    x = next(iter(dataset))

    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            x[k] = v.to(device)
    
    semantic_gt = {
        "semantic_positional_map_aug": x["semantic_positional_map_aug"][:,1],
        "semantic_speed_map_aug": x["semantic_speed_map_aug"][:,1],
        "semantic_class_map_aug": x["semantic_class_map_aug"][:,1],
    }
    
    with torch.no_grad():            
        out = model.net(**x)["semantic_bev"]
    
    aggr_gt = torch.zeros_like(semantic_gt["semantic_positional_map_aug"])
    for v in semantic_gt.values():
        aggr_gt += v
    aggr_gt = aggr_gt.float()/len(semantic_gt)
    
    import pdb; pdb.set_trace()
    
    loss = cos_similarity_loss(out, semantic_gt)
    metric = CosineSimilarity()
    
    metric.update(out, aggr_gt)

@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    pruebas(cfg)



if __name__ == "__main__":
    main()
