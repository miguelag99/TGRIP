"""Waymo: are the vehicles predicted outside the cameras field of view real or hallucinated?

Waymo has no rear camera, so vehicles outside every camera FOV are ignored by the loss and the
metrics and can not be judged there. This script scores the predictions against ALL ground-truth
vehicles (ignored ones included), separately inside and outside the FOV of the present cameras:
  - pixel precision / recall / IoU,
  - blob precision: fraction of predicted connected components overlapping a GT vehicle.

Usage (same config as visualization; evaluates every `stride`-th validation sample):
    uv run tgrip/eval_waymo_fov.py ckpt.path=checkpoints/Waymo_finetune.ckpt +stride=10
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig
from pyquaternion import Quaternion
from scipy import ndimage
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tgrip import utils

log = utils.get_pylogger(__name__)

# Heights (m, ego frame) at which a BEV cell must project into a camera to be inside its FOV.
FOV_HEIGHTS = [0.5, 1.5]
MIN_DEPTH = 0.5


def bev_fov_mask(nusc, rec: dict, grid: DictConfig, img_size) -> np.ndarray:
    """(H, W) bool mask of the BEV cells seen by at least one camera of the record.
    BEV arrays are rasterized with x (forward) decreasing along rows and y (left) decreasing
    along columns: front up, left on the left."""
    xs = (np.arange(*grid.xbound) + grid.xbound[2] / 2)[::-1]
    ys = (np.arange(*grid.ybound) + grid.ybound[2] / 2)[::-1]
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    W, H = img_size
    mask = np.zeros(yy.shape, dtype=bool)
    for cam, sd_token in rec["data"].items():
        if cam == "LIDAR_TOP":
            continue
        cs = nusc.get("calibrated_sensor", nusc.get("sample_data", sd_token)["calibrated_sensor_token"])
        R, t, K = Quaternion(cs["rotation"]).rotation_matrix, np.array(cs["translation"]), np.array(cs["camera_intrinsic"])
        for z in FOV_HEIGHTS:
            pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
            pts = R.T @ (pts - t[:, None])
            depth = pts[2]
            uv = (K @ pts)[:2] / np.maximum(depth, 1e-6)
            seen = (depth > MIN_DEPTH) & (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
            mask |= seen.reshape(mask.shape)
    return mask


@utils.task_wrapper
def evaluate(cfg: DictConfig):
    cfg.data.prefetch_factor = None
    cfg.data.num_workers = 0
    cfg.data.batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    ds = datamodule.val_dataloader().dataset
    assert cfg.data.is_waymo, "Only meaningful on Waymo (no rear camera)."

    model = hydra.utils.instantiate(cfg.model)
    model = utils.load_state_model(
        model, utils.get_ckpt_from_path(cfg.ckpt.path), cfg.ckpt.model.freeze, cfg.ckpt.model.load
    ).to(device).eval()

    img_size = (cfg.data.img_params.W, cfg.data.img_params.H)
    stride = cfg.get("stride", 10)
    T = len(cfg.data.bev_T_P)
    # Counters per region, per BEV frame.
    keys = ["tp", "pred", "gt", "blobs", "blobs_real", "ign_px", "ign_px_out", "used_px", "used_px_in"]
    stats = {r: {k: np.zeros(T) for k in keys} for r in ["in_fov", "out_fov"]}

    for idx in tqdm(range(0, len(ds), stride)):
        rec = ds.ixes[ds.indices[idx][ds.present_index[0]]]
        fov = bev_fov_mask(ds.nusc, rec, cfg.data.grid, img_size)

        x = ds[idx]
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device).unsqueeze(0)
        with torch.inference_mode():
            # Channels: (background, vehicle), as in the IoU metrics.
            pred = (model(x)["bev"]["binimg"][0].argmax(1) == 1).cpu().numpy()

        gt_all = x["binimg"][0, :, 0].cpu().numpy() > 0  # Every vehicle, ignored or not.
        valid = x["valid_binimg"][0, :, 0].cpu().numpy()

        for t in range(T):
            for region, m in [("in_fov", fov), ("out_fov", ~fov)]:
                s, p, g = stats[region], pred[t] & m, gt_all[t] & m
                s["tp"][t] += (p & g).sum()
                s["pred"][t] += p.sum()
                s["gt"][t] += g.sum()
                blobs, n = ndimage.label(p)
                s["blobs"][t] += n
                s["blobs_real"][t] += len(np.unique(blobs[g & (blobs > 0)]))
            # Sanity check of the FOV mask: ignored vehicles should be outside, used ones inside.
            s = stats["out_fov"]
            s["ign_px"][t] += (gt_all[t] & ~valid[t]).sum()
            s["ign_px_out"][t] += (gt_all[t] & ~valid[t] & ~fov).sum()
            s["used_px"][t] += (gt_all[t] & valid[t]).sum()
            s["used_px_in"][t] += (gt_all[t] & valid[t] & fov).sum()

    times = [int(t) for t, _ in cfg.data.bev_T_P]
    results = {"stride": stride, "samples": len(range(0, len(ds), stride)), "bev_T": times}
    div = lambda a, b: np.round(100 * a / np.maximum(b, 1), 2).tolist()
    for region, s in stats.items():
        results[region] = {
            "pixel_precision": div(s["tp"], s["pred"]),
            "pixel_recall": div(s["tp"], s["gt"]),
            "pixel_iou": div(s["tp"], s["pred"] + s["gt"] - s["tp"]),
            "blob_precision": div(s["blobs_real"], s["blobs"]),
            "pred_blobs": s["blobs"].astype(int).tolist(),
        }
    s = stats["out_fov"]
    results["fov_mask_check"] = {
        "ignored_vehicle_px_outside_fov_%": div(s["ign_px_out"], s["ign_px"]),
        "used_vehicle_px_inside_fov_%": div(s["used_px_in"], s["used_px"]),
    }

    log.info(json.dumps(results, indent=2))
    out = Path(cfg.paths.output_dir) / "fov_eval.json"
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Saved to {out}")
    return results, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="visualize.yaml")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
