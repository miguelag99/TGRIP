import hydra
import pyrootutils
import torch
import numpy as np
import os

from transformers import (
    pipeline,
    Sam3Processor,
    Sam3Model,
    CLIPProcessor,
    CLIPModel,
)
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from einops import rearrange
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Optional
from tqdm import tqdm
from safetensors.torch import save_file, load_file
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
def generate_semantic_bev(cfg: DictConfig) -> None:
    """This functions generates the corresponding semantic maps in the
    different image planes. SAM3 localizes the instances and CLIP-V generates
    the text aligned feature to fill the map.

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
        
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    
    # CLIP model
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
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
            
            img = Image.open(img_path).convert("RGB")
            
            img_crops = []
            for gt in gt_obj:
                corners_img = view_points(gt.corners(), intrinsic, normalize=True)[:2, :]
                corners = corners_img.T  # shape (8, 2)
            
                # Get 2D bounding box from projected corners
                x_min = max(0, corners[:, 0].min())
                y_min = max(0, corners[:, 1].min())
                x_max = min(img.width, corners[:, 0].max())
                y_max = min(img.height, corners[:, 1].max())
                
                # Crop and resize to square
                w, h = x_max - x_min, y_max - y_min
                size = max(h, w)
                crop = img.crop((x_min, y_min, x_max, y_max))
                if h != w:
                    crop = crop.resize((int(size), int(size)), Image.NEAREST)
                img_crops.append(crop)
            
            if len(img_crops) > 0:
                # Get CLIP visual features
                clip_inputs = clip_processor(
                    images=img_crops,
                    return_tensors="pt",
                    padding=True,
                    do_center_crop=False,
                    do_resize=True,
                ).to(device)
                clip_features = clip.get_image_features(**clip_inputs)  # (1, 512)
                clip_features = clip_features / clip_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                
            for idx, (gt, crop) in enumerate(zip(gt_obj, img_crops)):
                current_area = crop.width * crop.height
                if gt.token in crops_info:
                    # If the object is in multiple cameras, keep the largest crop
                    if current_area > crops_info[gt.token]["area"]:
                        batch_data[gt.token] = clip_features[idx].cpu().contiguous()
                        crops_info[gt.token] = {
                            "cam_idx": i,
                            "area": current_area,
                            "bbox": (x_min, y_min, x_max, y_max)
                        }
                else:
                    batch_data[gt.token] = clip_features[idx].cpu().contiguous()
                    crops_info[gt.token] = {
                        "cam_idx": i,
                        "area": current_area,
                        "bbox": (x_min, y_min, x_max, y_max)
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
            
            img = Image.open(img_path).convert("RGB")
            
            img_crops = []
            for gt in gt_obj:
                corners_img = view_points(gt.corners(), intrinsic, normalize=True)[:2, :]
                corners = corners_img.T  # shape (8, 2)
            
                # Get 2D bounding box from projected corners
                x_min = max(0, corners[:, 0].min())
                y_min = max(0, corners[:, 1].min())
                x_max = min(img.width, corners[:, 0].max())
                y_max = min(img.height, corners[:, 1].max())
                
                # Crop and resize to square
                w, h = x_max - x_min, y_max - y_min
                size = max(h, w)
                crop = img.crop((x_min, y_min, x_max, y_max))
                if h != w:
                    crop = crop.resize((int(size), int(size)), Image.NEAREST)
                img_crops.append(crop)
            
            if len(img_crops) > 0:
                # Get CLIP visual features
                clip_inputs = clip_processor(
                    images=img_crops,
                    return_tensors="pt",
                    padding=True,
                    do_center_crop=False,
                    do_resize=True,
                ).to(device)
                clip_features = clip.get_image_features(**clip_inputs)  # (1, 512)
                clip_features = clip_features / clip_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                
            for idx, (gt, crop) in enumerate(zip(gt_obj, img_crops)):
                current_area = crop.width * crop.height
                if gt.token in crops_info:
                    # If the object is in multiple cameras, keep the largest crop
                    if current_area > crops_info[gt.token]["area"]:
                        batch_data[gt.token] = clip_features[idx].cpu().contiguous()
                        crops_info[gt.token] = {
                            "cam_idx": i,
                            "area": current_area,
                            "bbox": (x_min, y_min, x_max, y_max)
                        }
                else:
                    batch_data[gt.token] = clip_features[idx].cpu().contiguous()
                    crops_info[gt.token] = {
                        "cam_idx": i,
                        "area": current_area,
                        "bbox": (x_min, y_min, x_max, y_max)
                    }
                    
        # Save processed data for sample and image
        filename = f"{dataset.dataset.semanticroot}/semantic_data_{scene_token['token']}.safetensors"          
        save_executor.submit(save_file, batch_data, filename)
    
    
    print("Processing complete. Waiting for pending saves...")
    save_executor.shutdown(wait=True)
    print("All files saved.")


def get_sam_masks(model, processor, image, text_prompt, device):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    return results

@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    generate_semantic_bev(cfg)

if __name__ == "__main__":
    main()
