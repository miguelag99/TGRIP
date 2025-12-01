import hydra
import matplotlib.pyplot as plt
import pyrootutils
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
from einops import rearrange
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Optional
import io
from PIL import Image

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tgrip import utils
from tgrip.utils.imgs import NORMALIZE_IMG
from tgrip.visualize_batch import generate_gt_instance_pred
from torchvision import transforms

log = utils.get_pylogger(__name__)


CLASS_CONDITIONS = {
    "vehicle.bicycle": {"text": "Bicycle", "idx": 1},
    "vehicle.bus.bendy": {"text": "Bus", "idx": 2},
    "vehicle.bus.rigid": {"text": "Bus", "idx": 3},
    "vehicle.car": {"text": "Car", "idx": 4},
    "vehicle.construction": {"text": "Construction Vehicle", "idx": 5},
    "vehicle.emergency.ambulance": {"text": "Ambulance", "idx": 6},
    "vehicle.emergency.police": {"text": "Police Car", "idx": 7},
    "vehicle.motorcycle": {"text": "Motorcycle", "idx": 8},
    "vehicle.trailer": {"text": "Trailer", "idx": 9},
    "vehicle.truck": {"text": "Truck", "idx": 10},
}


def _fill_semantic_maps(batch, bs, dataset):
    """Fill semantic maps with true CLIP embeds instead of indices."""
    bev_h, bev_w = batch["semantic_map"].shape[-2:]
    tout = batch["semantic_map"].shape[1]
    text_dim = 512
    device = batch["semantic_map"].device

    base_shape = (bs, tout, text_dim, bev_h, bev_w)
    dtype = torch.float32
    final_semantics = torch.zeros(base_shape, device=device, dtype=dtype)
    final_semantics_aug = torch.zeros_like(final_semantics)
    
    conditions = getattr(dataset, 'class_conditions')
    for k, v in conditions.items():
        idx = v["idx"]
        embedding = v["embedding"].to(dtype).to(device)
        embedding_expanded = embedding.view(1, 1, -1, 1, 1)

        mask = (batch["semantic_map"] == idx).to(device)
        mask_aug = (batch["semantic_map_aug"] == idx).to(device)
        mask_expanded = mask.expand(-1, -1, embedding_expanded.shape[2], -1, -1)
        mask_aug_expanded = mask_aug.expand_as(mask_expanded)

        final_semantics = torch.where(mask_expanded, embedding_expanded, final_semantics)
        final_semantics_aug = torch.where(mask_aug_expanded, embedding_expanded, final_semantics_aug)

    batch['semantic_map'] = final_semantics
    batch['semantic_map_aug'] = final_semantics_aug


class ModelVisualizer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize model
        log.info(f"Instantiating model <{cfg.model._target_}>")
        ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        self.model = utils.load_state_model(
            self.model, ckpt, cfg.ckpt.model.freeze, 
            cfg.ckpt.model.load, verbose=1
        ).to(self.device)
        self.model.eval()
        
        # Initialize datamodule
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        cfg.data.prefetch_factor = None
        cfg.data.num_workers = 0
        cfg.data.batch_size = 1
        cfg.data.normalize_img = True
        
        self.datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
        self.datamodule.setup()
        
        self.dataloaders = {
            'train': self.datamodule.train_dataloader(),
            'val': self.datamodule.val_dataloader(),
            'test': self.datamodule.test_dataloader() if hasattr(self.datamodule, 'test_dataloader') else None
        }
    
    def visualize_sample(self, sample_idx: int, split: str, class_name: str):
        """Visualize a specific sample from the dataset."""
        
        # Get appropriate dataloader
        dataloader = self.dataloaders.get(split)
        if dataloader is None:
            return None, f"Split '{split}' not available"
        
        # Get sample
        try:
            dataset = dataloader.dataset
            if sample_idx >= len(dataset):
                return None, f"Sample index {sample_idx} out of range (max: {len(dataset)-1})"
            
            x = dataset[sample_idx]
            
            # Add batch dimension and move to device
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                out = self.model.net(**x)
            
            # Fill semantic maps
            _fill_semantic_maps(x, 1, dataset)
            
            # Extract dimensions
            B, T, C, H, W = x['semantic_map'].shape
            
            timestamp = 0 # Semantic maps are single timestamp
            if timestamp >= T:
                return None, f"Timestamp {timestamp} out of range (max: {T-1})"
            
            # PCA visualization
            vis_gt_semantic = utils.extract_pca_features(
                x['semantic_map_aug'][:, timestamp], 3
            )[0]
            
            vis_semantic = utils.extract_pca_features(
                out['semantic']['semantic_bev'][:, 0], 3
            )[0]
            
            # Cosine similarity maps
            text_embed = self.model.text_encoder(class_name).to(self.device)
            
            cosine_map_gt = F.cosine_similarity(
                F.normalize(text_embed, dim=1),
                F.normalize(rearrange(x['semantic_map_aug'][0, timestamp], 'c h w -> (h w) c'), dim=1),
                dim=1
            ).view(H, W)
            
            cosine_map_out = F.cosine_similarity(
                F.normalize(text_embed, dim=1),
                F.normalize(rearrange(out['semantic']['semantic_bev'][0, 0], 'c h w -> (h w) c'), dim=1),
                dim=1
            ).view(H, W)
            
            # Prepare camera images
            # x['imgs'] shape: (B, T, num_cams, C, H, W)
            imgs = x['imgs'][0, timestamp]  # (num_cams, C, H, W)
            num_cams = imgs.shape[0]
            
            # Denormalize images for visualization
            mean = torch.tensor(NORMALIZE_IMG.transforms[1].mean).view(1, 3, 1, 1).to(imgs.device)
            std = torch.tensor(NORMALIZE_IMG.transforms[1].std).view(1, 3, 1, 1).to(imgs.device)
            imgs_denorm = imgs * std + mean
            imgs_denorm = torch.clamp(imgs_denorm, 0, 1)
            
            # Create plot: camera images on top, semantic visualizations on bottom
            fig = plt.figure(figsize=(24, 12))
            gs = fig.add_gridspec(2, max(num_cams, 4), height_ratios=[1, 1])
            
            # Top row: Camera images
            for i in range(num_cams):
                ax = fig.add_subplot(gs[0, i])
                img_np = imgs_denorm[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img_np)
                ax.axis('off')
                ax.set_title(f"Camera {i}", fontsize=10)
            
            # Bottom row: Semantic visualizations (centered if fewer than num_cams)
            offset = max(0, (num_cams - 4) // 2)
            
            ax0 = fig.add_subplot(gs[1, offset])
            ax0.imshow(vis_gt_semantic.permute(1, 2, 0).cpu().numpy())
            ax0.axis('off')
            ax0.set_title("PCA of Semantic Class Map", fontsize=12)
            
            ax1 = fig.add_subplot(gs[1, offset + 1])
            im1 = ax1.imshow(cosine_map_gt.cpu().numpy(), cmap='viridis', vmin=0.25, vmax=1)
            ax1.axis('off')
            ax1.set_title(f"Cosine Similarity Map (GT) - {class_name}", fontsize=12)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            ax2 = fig.add_subplot(gs[1, offset + 2])
            ax2.imshow(vis_semantic.permute(1, 2, 0).cpu().numpy())
            ax2.axis('off')
            ax2.set_title("PCA of Semantic BEV Features", fontsize=12)
            
            ax3 = fig.add_subplot(gs[1, offset + 3])
            im2 = ax3.imshow(cosine_map_out.cpu().numpy(), cmap='viridis', vmin=0.25, vmax=1)
            ax3.axis('off')
            ax3.set_title(f"Cosine Similarity Map (Output) - {class_name}", fontsize=12)
            plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Convert to PIL Image for Gradio
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img, f"Successfully visualized sample {sample_idx} from {split} split (timestamp={timestamp}, {num_cams} cameras)"
            
        except Exception as e:
            return None, f"Error: {str(e)}"


def create_gradio_interface(cfg: DictConfig):
    """Create Gradio interface for model visualization."""
    
    visualizer = ModelVisualizer(cfg)
    
    # Get available classes
    class_names = [v["text"] for v in CLASS_CONDITIONS.values()]
    
    with gr.Blocks(title="Model Visualization") as demo:
        gr.Markdown("# Model Semantic Visualization")
        gr.Markdown("Visualize semantic maps and cosine similarity for different samples")
        
        with gr.Row():
            with gr.Column(scale=1):
                sample_idx = gr.Number(
                    label="Sample Index", 
                    value=0, 
                    precision=0,
                    minimum=0
                )
                split = gr.Dropdown(
                    label="Dataset Split",
                    choices=["train", "val", "test"],
                    value="val"
                )
                class_name = gr.Dropdown(
                    label="Class Name",
                    choices=class_names,
                    value="Car"
                )
                btn = gr.Button("Visualize", variant="primary")
            
            with gr.Column(scale=3):
                output_img = gr.Image(label="Visualization", type="pil")
                status = gr.Textbox(label="Status", interactive=False)
        
        btn.click(
            fn=visualizer.visualize_sample,
            inputs=[sample_idx, split, class_name],
            outputs=[output_img, status]
        )
        
        # Auto-load first sample on launch
        demo.load(
            fn=visualizer.visualize_sample,
            inputs=[sample_idx, split, class_name],
            outputs=[output_img, status]
        )
    
    return demo


@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    demo = create_gradio_interface(cfg)
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()