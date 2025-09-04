import os
import time
import numpy as np
import sys
import matplotlib.pyplot as plt


from pathlib import Path
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

import hydra
import pyrootutils
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from tgrip import utils
from tgrip.models.text_encoder import CLIPEncoder, BertMiniEncoder
from tgrip.models.text_conditioner import TextCrossAttention, MultiFiLMModulation
from pathlib import Path
import numpy as np

log = utils.get_pylogger(__name__)




def extract_pca_features(
    features: torch.Tensor,
    n_components: int = 3,
) -> torch.Tensor:
    # Ensure the input is on CPU and converted to numpy
    B, C, H, W = features.shape
    
    # Reshape features to (Batch, Channels, H*W)
    projected_heatmap = features.view(B, C, -1).cpu().numpy()
    
    # Initialize output list
    output = []
    
    # Process each batch item
    for heatmap in projected_heatmap:
        # Transpose to make it (H*W, Channels) for PCA
        heatmap_transposed = heatmap.T
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(heatmap_transposed)
        
        # Normalize to 0-255 range
        pca_features = minmax_scale(pca_features, feature_range=(0, 255)).astype(np.uint8)
        
        # Reshape back to (n_components, H, W)
        pca_features = pca_features.T.reshape(n_components, H, W)
        
        # Convert to torch tensor
        pca_features = torch.from_numpy(pca_features)
        output.append(pca_features)
    
    # Stack the batch
    pca_features = torch.stack(output, dim=0)
    
    return pca_features


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
    dataset = datamodule.val_dataloader()
    
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
    
    with torch.no_grad():
        output = model.net(**x)['bev']




@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    utils.modif_config_based_on_flags(cfg)
    pruebas(cfg)



if __name__ == "__main__":
    main()
