"""
Visualize GT vs. predicted instance segmentation across all future timesteps.

Saves one figure per scene with two rows:
  Row 0: GT Instance Segmentation  t=1 | t=2 | … | t=N
  Row 1: Out Instance Segmentation t=1 | t=2 | … | t=N

GT maps come directly from the dataloader (batch["instance"]).
Predicted maps are produced by predict_instance_segmentation.

Usage (inside Docker container):
    uv run scripts/visualize_instances_over_time.py ++scene_ids=[0,1,2]
    uv run scripts/visualize_instances_over_time.py ++scene_ids=[25] ckpt.path=checkpoints/my.ckpt
"""

import hydra
import pyrootutils
import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L


from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tgrip import utils

log = utils.get_pylogger(__name__)


def visualise(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.data.prefetch_factor = None
    cfg.data.num_workers = 0
    cfg.data.batch_size = 1
    cfg.data.normalize_img = True

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    from pytorch_lightning import LightningDataModule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataset = datamodule.train_dataloader().dataset

    ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    from pytorch_lightning import LightningModule
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = utils.load_state_model(
        model,
        ckpt,
        cfg.ckpt.model.freeze,
        cfg.ckpt.model.load,
        verbose=1,
    ).to(device)
    model.eval()

    postproc = cfg.model.get("postproc_kwargs", {})
    spatial_extent = (cfg.data.grid.xbound[1], cfg.data.grid.ybound[1])
    conf_threshold = postproc.get("conf_threshold", 0.1)
    nms_kernel_size = postproc.get("nms_kernel_size", None)
    
    output_dir = Path(cfg.paths.output_dir).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = OmegaConf.to_container(cfg.get("scene_ids", [0]))

    for scene_id in scene_ids:
        sample = dataset[scene_id]

        # GT instance maps — shape [T, 1, H, W]; skip T=0 (present frame)
        gt_instance = sample["instance_aug"][1:].squeeze(1)  # [T_future, H, W]

        # Move sample to device for model inference
        batch = {
            k: v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }

        with torch.inference_mode():
            output = model(batch)

        with_centerness = "centerness" in output.get("bev", {}) and output["bev"]["centerness"] is not None
        pred_data = {
            "instance_flow": output["bev"]["flow"],
            "segmentation": output["bev"]["binimg"],
            "centerness": output["bev"]["centerness"] if with_centerness else None,
        }
        pred_instance_seg = utils.predict_instance_segmentation(
            pred_data,
            spatial_extent=spatial_extent,
            conf_threshold=conf_threshold,
            nms_kernel_size=nms_kernel_size,
        )
        pred_instance = pred_instance_seg[0, 1:].cpu()  # [T_future, H, W]
        
        n_future = min(gt_instance.shape[0], pred_instance.shape[0])
        fig, axes = plt.subplots(3, n_future, figsize=(4 * n_future, 8))
        if n_future == 1:
            axes = axes.reshape(3, 1)

        for t in range(n_future):
            gt_frame = gt_instance[t].cpu().numpy()
            
            unique_gt = np.unique(gt_frame)
            unique_gt = unique_gt[unique_gt > 0]
            instance_map_gt = {int(i): int(i) for i in unique_gt}
            gt_img = utils.plot_instance_map(gt_frame, instance_map_gt)
            
            unique_pred_old = np.unique(pred_instance[t].numpy())
            unique_pred_old = unique_pred_old[unique_pred_old > 0]
            instance_map_pred_old = {int(i): int(i) for i in unique_pred_old}
            pred_img_old = utils.plot_instance_map(pred_instance[t].numpy(), instance_map_pred_old)

            axes[0, t].imshow(gt_img)
            axes[0, t].set_title(f"Gt Instance Segmentation t={t + 1}", fontsize=9)
            axes[0, t].axis("off")
            
            axes[1, t].imshow(pred_img_old)
            axes[1, t].set_title(f"Out Instance Segmentation t={t + 1}", fontsize=9)
            axes[1, t].axis("off")
            
            axes[2, t].imshow(output["bev"]["centerness"][0, t+1, 0].cpu(), vmin=0, vmax=1, cmap="viridis")
            axes[2, t].set_title(f"Centerness t={t + 1}", fontsize=9)
            axes[2, t].axis("off")

        plt.tight_layout()
        out_path = output_dir / f"instances_scene_{scene_id}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved: {out_path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    visualise(cfg)


if __name__ == "__main__":
    main()
