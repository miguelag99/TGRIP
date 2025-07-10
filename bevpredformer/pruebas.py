import os
import time

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from bevpredformer import utils


log = utils.get_pylogger(__name__)


class CLIPEncoder(torch.nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', freeze=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.encoder.device)
        out = self.encoder(**tokens)
        return out.pooler_output

class BertMiniEncoder(torch.nn.Module):
    def __init__(self, pooling='mean', freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
        self.encoder = AutoModel.from_pretrained("prajjwal1/bert-mini")
        self.pooling = pooling
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.encoder.device)
        out = self.encoder(**tokens)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0, :]
        else:
            mask = tokens['attention_mask'].unsqueeze(-1)
            x = out.last_hidden_state * mask
            return x.sum(dim=1) / mask.sum(dim=1)


def pruebas(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg.data.prefetch_factor = 0
    cfg.data.num_workers = 1
    cfg.data.batch_size = 1
    cfg.data.normalize_img = False
    cfg.data.version = 'mini'

    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataset = datamodule.val_dataloader().dataset
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)

    model = hydra.utils.instantiate(cfg.model)
    model = utils.load_state_model(
        model,
        ckpt,
        cfg.ckpt.model.freeze,
        cfg.ckpt.model.load,
        verbose=1,
    )

    x = dataset[0]
    
    text_encoder = CLIPEncoder()
    text_encoder.eval()
    
    import pdb; pdb.set_trace()
    


    
    
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    utils.modif_config_based_on_flags(cfg)
    pruebas(cfg)



if __name__ == "__main__":
    main()
