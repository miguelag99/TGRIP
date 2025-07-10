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
from pytorch_lightning import LightningDataModule, Trainer, Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.profiler import PyTorchProfiler

import matplotlib.pyplot as plt


from torch.profiler import ProfilerActivity

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from bevpredformer import utils
import pickle

log = utils.get_pylogger(__name__)

def train(cfg: DictConfig) -> Tuple[dict, dict]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    cfg.data.version = 'mini'
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataloader = datamodule.val_dataloader()
    iter_dataloader = iter(dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model).to(device)
    
    def move_to_device(batch, device):
        """Move batch of inputs to device if they are tensor."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [move_to_device(v, device) for v in batch]
        elif isinstance(batch, tuple):
            return tuple(move_to_device(v, device) for v in batch)
        else:
            return batch

    # Get a batch and move it to GPU
    batch = next(iter_dataloader)
    batch = move_to_device(batch, device)
    
    model.eval()
    # Warm up the model
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch)
    
    with torch.no_grad():
        # Run the model on the batch multiple times and compute average
        num_samples = 10
        elapsed_times = []
        
        for _ in range(num_samples):
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            output = model(batch)
            t2.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded
            elapsed_times.append(t1.elapsed_time(t2))
        
        avg_elapsed_time = sum(elapsed_times) / num_samples
        log.info(f"Average elapsed time for model inference over {num_samples} runs: {avg_elapsed_time:.2f} ms")
        log.info(f"Standard deviation: {torch.tensor(elapsed_times).std().item():.2f} ms")
                
    
    return 0, 0
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.modif_config_based_on_flags(cfg)
    
    utils.extras(cfg)

    train(cfg)



if __name__ == "__main__":
    main()
