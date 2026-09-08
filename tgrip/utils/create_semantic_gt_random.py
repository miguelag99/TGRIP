import hydra
import pyrootutils
import torch
import os

from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Optional
from tqdm import tqdm
from safetensors.torch import save_file
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tgrip import utils

log = utils.get_pylogger(__name__)
save_executor = ThreadPoolExecutor(max_workers=1)

torch.set_printoptions(precision=2, sci_mode=False)

CAM_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

def async_save(tensors, filename):
    """Background task to save file"""
    try:
        save_file(tensors, filename)
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

@torch.no_grad()
def generate_semantic_embeds(cfg: DictConfig) -> None:
    """Randomized-teacher control: generates a random (content-free) semantic
    embed for each object in each frame, in place of a real visual encoder.

    Object visibility/membership (which objects get an embed vs. fall back to
    the background embed) is computed exactly as in the other
    create_semantic_gt_*.py scripts, so this is a drop-in swap that isolates
    semantic content as the only difference. If training with these random
    embeds collapses performance, the gain from real embeds is due to
    semantic content and not just multi-task regularization.

    Args:
        cfg (DictConfig): Hydra config object.
    """

    cfg.data.prefetch_factor = None
    cfg.data.num_workers = 0
    cfg.data.batch_size = 1

    # Remove data augmentations
    cfg.data.normalize_img = False
    cfg.data.img_params.zoom_lim = [1.0, 1.0]
    cfg.data.img_params.rot_lim = [0.0, 0.0]

    cfg.data.img_params.min_visibility = 2  # Remove boxes with not enough visibility

    cfg.data.version = 'trainval'
    cfg.data.train_shuffle = False

    # Only load present info
    cfg.data.cam_T_P = [[0,0]]
    cfg.data.bev_T_P = [[0,0]]
    cfg.data.keep_input_detection = True
    cfg.data.keep_input_binimg = False
    cfg.data.keep_input_centr_offs = False
    cfg.data.keep_input_sampling = False
    cfg.data.keep_input_flow_map = False
    cfg.data.keep_input_semantic_maps = False

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    # Embedding dim must match the configured text encoder's output dim, so
    # random embeds are a drop-in replacement for the real visual embeds.
    embed_dim = hydra.utils.instantiate(cfg.model.text_encoder).out_dim

    seed = cfg.get("seed", None)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    def random_embeds(n):
        embeds = torch.randn(n, embed_dim, generator=generator)
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)

    # Generate train samples
    dataset = datamodule.train_dataloader()

    # Create folder if not exists
    os.makedirs(dataset.dataset.semanticroot, exist_ok=True)

    for batch in tqdm(dataset):
        batch_data = {}
        crops_info = {}
        assert batch["imgs"].shape[0] == 1  # batch size 1
        scene_token = batch["sample_tokens"][0][0] # Only the present frame

        for i, cam_name in enumerate(CAM_NAMES):

            img_path, gt_obj, intrinsic = dataset.dataset.nusc.get_sample_data(
                batch['sample_tokens'][0][0]['data'][cam_name],
                box_vis_level=BoxVisibility.ANY,
                selected_anntokens=batch['obj_tokens'][0][0]
            )

            img = Image.open(img_path)

            areas = []
            for gt in gt_obj:
                corners_img = view_points(gt.corners(), intrinsic, normalize=True)[:2, :]
                corners = corners_img.T  # shape (8, 2)

                # Get 2D bounding box from projected corners
                x_min = max(0, corners[:, 0].min())
                y_min = max(0, corners[:, 1].min())
                x_max = min(img.width, corners[:, 0].max())
                y_max = min(img.height, corners[:, 1].max())

                # Same square-crop size used for the tie-break area as in the
                # real visual encoders' scripts.
                w, h = x_max - x_min, y_max - y_min
                size = max(h, w)
                areas.append(size * size)

            if len(gt_obj) > 0:
                random_features = random_embeds(len(gt_obj))

            for idx, (gt, area) in enumerate(zip(gt_obj, areas)):
                current_area = area
                if gt.token in crops_info:
                    # If the object is in multiple cameras, keep the largest crop
                    if current_area > crops_info[gt.token]["area"]:
                        batch_data[gt.token] = random_features[idx].cpu().contiguous()
                        crops_info[gt.token] = {
                            "cam_idx": i,
                            "area": current_area,
                        }
                else:
                    batch_data[gt.token] = random_features[idx].cpu().contiguous()
                    crops_info[gt.token] = {
                        "cam_idx": i,
                        "area": current_area,
                    }

        # Save processed data for sample and image
        filename = f"{dataset.dataset.semanticroot}/semantic_data_{scene_token['token']}.safetensors"
        save_executor.submit(save_file, batch_data, filename)

    # Generate val samples
    dataset = datamodule.val_dataloader()
    for batch in tqdm(dataset):
        batch_data = {}
        crops_info = {}
        assert batch["imgs"].shape[0] == 1  # batch size 1
        scene_token = batch["sample_tokens"][0][0] # Only the present frame

        for i, cam_name in enumerate(CAM_NAMES):

            img_path, gt_obj, intrinsic = dataset.dataset.nusc.get_sample_data(
                batch['sample_tokens'][0][0]['data'][cam_name],
                box_vis_level=BoxVisibility.ANY,
                selected_anntokens=batch['obj_tokens'][0][0]
            )

            img = Image.open(img_path)

            areas = []
            for gt in gt_obj:
                corners_img = view_points(gt.corners(), intrinsic, normalize=True)[:2, :]
                corners = corners_img.T  # shape (8, 2)

                # Get 2D bounding box from projected corners
                x_min = max(0, corners[:, 0].min())
                y_min = max(0, corners[:, 1].min())
                x_max = min(img.width, corners[:, 0].max())
                y_max = min(img.height, corners[:, 1].max())

                w, h = x_max - x_min, y_max - y_min
                size = max(h, w)
                areas.append(size * size)

            if len(gt_obj) > 0:
                random_features = random_embeds(len(gt_obj))

            for idx, (gt, area) in enumerate(zip(gt_obj, areas)):
                current_area = area
                if gt.token in crops_info:
                    # If the object is in multiple cameras, keep the largest crop
                    if current_area > crops_info[gt.token]["area"]:
                        batch_data[gt.token] = random_features[idx].cpu().contiguous()
                        crops_info[gt.token] = {
                            "cam_idx": i,
                            "area": current_area,
                        }
                else:
                    batch_data[gt.token] = random_features[idx].cpu().contiguous()
                    crops_info[gt.token] = {
                        "cam_idx": i,
                        "area": current_area,
                    }

        # Save processed data for sample and image
        filename = f"{dataset.dataset.semanticroot}/semantic_data_{scene_token['token']}.safetensors"
        save_executor.submit(save_file, batch_data, filename)


    print("Processing complete. Waiting for pending saves...")
    save_executor.shutdown(wait=True)
    print("All files saved.")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    generate_semantic_embeds(cfg)

if __name__ == "__main__":
    main()
