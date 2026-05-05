import hydra
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import motmetrics as mm
from omegaconf import DictConfig
from typing import Optional
from tqdm.auto import tqdm

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra.utils
from pytorch_lightning import LightningDataModule

from tgrip import utils
from torchmetrics.metric import Metric
from einops import rearrange
import pytorch_lightning as L
import torch.nn as nn
from typing import Dict


log = utils.get_pylogger(__name__)

def calculate_tracking_metrics(
    model_output: dict,
    ground_truth: dict,
    geomscaler,
    nusc    ,
    batch_id: int = 0,
    vehicles_id: int = 1,
    spatial_extent: list = None,
    distance_threshold: float = 2.0,
):
    """Calculate tracking metrics (MOTA, MOTP, frame count) and return centers for plotting.

    Args:
        model_output: Output dictionary from model containing 'segmentation' and 'instance_flow' keys.
        ground_truth: Dictionary with ground truth data including 'centers_aug', 'obj_tokens'.
        geomscaler: GeomScaler instance for coordinate transformations.
        nusc: NuScenes instance for annotation lookup.
        batch_id: Batch index to evaluate. Defaults to 0.
        vehicles_id: Index corresponding to vehicle class. Defaults to 1.
        spatial_extent: BEV range in each axis in meters. Defaults to grid bounds from dataset.
        distance_threshold: Maximum distance for matching GT to predictions (meters). Defaults to 2.0.

    Returns:
        acc.
    """
    if spatial_extent is None:
        spatial_extent = [50.0, 50.0]

    # Predicted instance segmentation
    preds = model_output['segmentation'].detach()
    preds = torch.argmax(preds, dim=2, keepdims=True)
    foreground_masks = preds.squeeze(2) == vehicles_id

    batch_size, seq_len = preds.shape[:2]
    pred_inst = []
    for b in range(batch_size):
        pred_inst_batch = utils.instance_pred.get_instance_segmentation_and_centers(
            torch.softmax(model_output['segmentation'], dim=2)[b, 0:1, vehicles_id].detach(),
            model_output['instance_flow'][b, 1:2].detach(),
            foreground_masks[b, 1:2].detach(),
            nms_kernel_size=round(350 / spatial_extent[0]),
        )
        pred_inst.append(pred_inst_batch)
    pred_inst = torch.stack(pred_inst).squeeze(2)

    consistent_instance_seg = []
    for b in range(batch_size):
        consistent_instance_seg.append(
            utils.instance_pred.make_instance_id_temporally_consecutive(
                pred_inst[b:b + 1],
                preds[b:b + 1, 1:],
                model_output['instance_flow'][b:b + 1, 1:].detach(),
            )
        )
    consistent_instance_seg = torch.cat(consistent_instance_seg, dim=0)
    consistent_instance_seg = torch.cat([torch.zeros_like(pred_inst), consistent_instance_seg], dim=1)

    # Generate matched centers for predictions
    matched_centers = {}
    _, seq_len, h, w = consistent_instance_seg.shape
    
    grid = torch.stack(torch.meshgrid(
        torch.arange(h, dtype=torch.float, device=preds.device),
        torch.arange(w, dtype=torch.float, device=preds.device),
        indexing='ij'
    ))

    for instance_id in torch.unique(consistent_instance_seg[0, 1])[1:].cpu().numpy():
        for t in range(seq_len):
            instance_mask = consistent_instance_seg[0, t] == instance_id
            if instance_mask.sum() > 0:
                matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                    grid[:, instance_mask].mean(dim=-1)
                ]
            elif t > 0:
                matched_centers[instance_id] = matched_centers.get(
                    instance_id, []
                ) + [torch.tensor([np.nan, np.nan], device=preds.device)]

    for key, value in matched_centers.items():
        matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1]

    # Shift prediction IDs to account for leading zero frame
    matched_centers = {k - 1: v for k, v in matched_centers.items()}

    # Ground truth centers
    gt_centers = {}
    centers_aug = ground_truth["centers_aug"][batch_id]
    obj_tokens = ground_truth["obj_tokens"][batch_id]

    for t in range(len(centers_aug) - 2):
        centers_aug[t + 2] = -centers_aug[t + 2].fliplr()
        for instance_id, ann_token in enumerate(obj_tokens[t + 2]):
            instance_token = nusc.get("sample_annotation", ann_token)["instance_token"]
            if instance_token not in gt_centers:
                gt_centers[instance_token] = np.full((int(len(centers_aug) - 2), 2), np.inf, dtype=float)

            gt_centers[instance_token][t] = (
                geomscaler.pts_from_img_to_spatial(
                    geomscaler.pts_from_scaled_to_img(centers_aug[t + 2][instance_id])
                ).cpu().numpy()
            )

    # Transform GT centers to spatial domain and reindex
    gt_centers = {integer_id: x for integer_id, x in enumerate(gt_centers.values())}

    for k, v in matched_centers.items():
        matched_centers[k] = geomscaler.pts_from_img_to_spatial(v)

    # Calculate distance matrix
    gt_ids = list(gt_centers.keys())
    pred_ids = list(matched_centers.keys())
    num_frames = seq_len - 1
    
    distance_matrix = np.full((num_frames, len(gt_ids), len(pred_ids)), np.inf)
    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):
            gt_center = gt_centers[gt_id]
            pred_center = matched_centers[pred_id]
            distance_matrix[:, i, j] = np.linalg.norm(gt_center - pred_center, axis=1)

    # Create MOTAccumulator and update frame by frame
    acc = mm.MOTAccumulator(auto_id=True)

    for t in range(num_frames):
        current_gt_ids = [gt_id for gt_id in gt_ids if not np.isnan(gt_centers[gt_id][t]).any()]
        current_pred_ids = [pred_id for pred_id in pred_ids if not np.isnan(matched_centers[pred_id][t]).any()]

        if len(current_gt_ids) > 0 and len(current_pred_ids) > 0:
            curr_cost = distance_matrix[t, current_gt_ids][:, current_pred_ids]
            curr_cost = np.where(curr_cost > distance_threshold, np.nan, curr_cost)

            acc.update(current_gt_ids, current_pred_ids, curr_cost)

    return acc


def run_tracking_metrics(cfg: DictConfig) -> None:
    cfg.seed = 115
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg.data.prefetch_factor = 1
    cfg.data.num_workers = 1
    cfg.data.batch_size = 1
    cfg.data.normalize_img = True
    cfg.data.version = 'trainval'
    cfg.data.img_params.min_visibility = 1
    cfg.data.keep_input_detection = True
    cfg.data.coeffs.bev_aug.trans_rot = [0., 0., 0., 0., 0., 0.]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataloader = datamodule.val_dataloader()

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
    model.eval()

    accumulators = []

    for x in tqdm(dataloader):

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        with torch.no_grad():
            out = model.net(**x)["bev"]

        batch_id = 0

        out['binimg'] = out['binimg'][0, 1:, :, :].unsqueeze(0)
        out['flow'] = out['flow'][0, 1:, :, :].unsqueeze(0)

        data = {
            'segmentation': out['binimg'],
            'instance_flow': out['flow'],
            'centerness': out['centerness']
        }

        spatial_extent = (cfg.data.grid.xbound[1], cfg.data.grid.ybound[1])
        batch_acc = calculate_tracking_metrics(
            data,
            x,
            model.geomscaler,
            dataloader.dataset.nusc,
            batch_id=batch_id,
            spatial_extent=spatial_extent,
        )
        accumulators.append(batch_acc)
        
    summary = mm.metrics.create().compute_many(
        accumulators,
        metrics=['num_frames', 'mota', 'motp', 'num_switches', 'num_fragmentations'],
        generate_overall=True
    )
    log.info(f"Tracking Metrics Summary for {cfg.ckpt.path}:\n{summary}")
    
@hydra.main(version_base="1.3", config_path="../../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    run_tracking_metrics(cfg)


if __name__ == "__main__":
    main()
