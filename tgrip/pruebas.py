import os
import time

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from tgrip import utils
from tgrip.models.text_encoder import CLIPEncoder, BertMiniEncoder
from tgrip.models.text_conditioner import TextCrossAttention, MultiFiLMModulation

log = utils.get_pylogger(__name__)




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
    # ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)

    model = hydra.utils.instantiate(cfg.model)
    # model = utils.load_state_model(
    #     model,
    #     ckpt,
    #     cfg.ckpt.model.freeze,
    #     cfg.ckpt.model.load,
    #     verbose=1,
    # ).to(device)

    x = next(iter(dataset))

    bev_feats = torch.randn(2, 128, 200, 200).to(device)  # Simulated BEV features

    text_encoder = CLIPEncoder()
    text_encoder.eval().to(device)
    bev_conditioner = TextCrossAttention(bev_dim=128, text_dim=512, patch_size=10, attn_heads=4)
    bev_conditioner.eval().to(device)
    
    text_feats = text_encoder(x['text_condition'])
    import pdb; pdb.set_trace()

    with torch.no_grad():
        # Apply FiLM modulation
        modulated_bev_feats = bev_conditioner(bev_feats, text_feats)
     
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    utils.modif_config_based_on_flags(cfg)
    pruebas(cfg)



if __name__ == "__main__":
    main()
