import hydra
import matplotlib
import matplotlib.pyplot as plt
import pyrootutils
import pytorch_lightning as L
import torch
import torch.nn as nn


from einops import rearrange
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from typing import  Optional

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tgrip import utils
from tgrip.visualize_batch import generate_gt_instance_pred


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
    dataset = datamodule.train_dataloader()

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
    
    imgs = x['imgs'] # b t 6 c h w
    out_seg = x['binimg'] # b t 1 h w
    out_seg_aug = x['binimg_aug'] # b t 1 h w
    out_sem_pos = x['semantic_positional_map'] # b t c h w
    out_sem_pos_aug = x['semantic_positional_map_aug'] # b t c h w
    out_sem_vel = x['semantic_speed_map'] # b t c h w
    out_sem_vel_aug = x['semantic_speed_map_aug'] # b t c h w
    out_sem_cls = x['semantic_class_map'] # b t c h w
    out_sem_cls_aug = x['semantic_class_map_aug'] # b t c h w

    t = 1
    fig, axs = plt.subplots(3, 6, figsize=(40, 20))

    # Binary segmentation
    axs[0, 0].imshow(out_seg[0, t, 0].cpu().numpy(), cmap='gray')
    axs[0, 0].set_title('Segmentation')
    axs[0, 0].axis('off')

    # Augmented binary segmentation
    axs[0, 1].imshow(out_seg_aug[0, t, 0].cpu().numpy(), cmap='gray')
    axs[0, 1].set_title('Augmented Segmentation')
    axs[0, 1].axis('off')

    # Semantic positional map (PCA)
    sem_pos_pca = utils.extract_pca_features(out_sem_pos[:, t].cpu())
    axs[0, 2].imshow(sem_pos_pca[0].permute(1, 2, 0), cmap='viridis')
    axs[0, 2].set_title('Semantic Positional')
    axs[0, 2].axis('off')

    # Augmented semantic positional map (PCA)
    sem_pos_aug_pca = utils.extract_pca_features(out_sem_pos_aug[:, t].cpu())
    axs[0, 3].imshow(sem_pos_aug_pca[0].permute(1, 2, 0), cmap='viridis')
    axs[0, 3].set_title('Augmented Semantic Positional')
    axs[0, 3].axis('off')
    
    # Extract cosine similarity between text_embed and bev_semantic_map
    cos = nn.CosineSimilarity(dim=-1)
    semantic_score_front = cos(
        out_sem_pos_aug[0, t].permute(1, 2, 0),
        model.net.text_encoder(["front"])
    )
    vmin, vmax = 0.0, 1.0

    # Similarity with different texts
    semantic_score_front = cos(
        out_sem_pos_aug[0, t].permute(1, 2, 0),
        model.net.text_encoder(["front"])
    )
    im_front = axs[0, 4].imshow(semantic_score_front.cpu().numpy(), cmap='coolwarm',
                                vmin=vmin, vmax=vmax)
    axs[0, 4].set_title('Sim. with front')
    axs[0, 4].axis('off')

    # Similarity with different texts
    semantic_score_back_right = cos(
        out_sem_pos_aug[0, t].permute(1, 2, 0),
        model.net.text_encoder(["back_right"])
    )
    im_back_right = axs[0, 5].imshow(semantic_score_back_right.cpu().numpy(), cmap='coolwarm',
                                     vmin=vmin, vmax=vmax)
    axs[0, 5].set_title('Sim. with back_right')
    axs[0, 5].axis('off')

    # Common colorbar for all similarities
    fig.colorbar(
        im_front,
        ax=axs[0, 4:7],
        fraction=0.046,
        pad=0.04,
        label="Cosine Similarity (0-1)",
    )

    # GT
    axs[1, 0].imshow(
        generate_gt_instance_pred(
            cfg.data.grid,
            batch={
                "binimg_aug": x["binimg_aug"][0],
                "flow_map_aug": x["flow_map_aug"][0],
                "centerness_aug": x["centerness_aug"][0],
            },
            plot_ego=False,
        )
    )
    axs[1, 0].set_title('Instance pred gt')
    axs[1, 0].axis('off')

    # Semantic velocity map
    b, _, c, h, w = out_sem_vel.shape
    vis_img = torch.zeros((h * w, 3))
    rearranged = rearrange(out_sem_vel[0, t], "c h w -> (h w) c").cpu()
    valid_mask = rearranged[:, 0] != 0

    unique_vals, inverse_indices = torch.unique(rearranged, dim=0, return_inverse=True)
    colors = matplotlib.colormaps['tab20'].resampled(unique_vals.shape[0])
    for idx, color in enumerate(colors.colors):
        mask = (inverse_indices == idx)
        # If the unique value is all zeros, plot as black
        if torch.all(unique_vals[idx] == 0):
            vis_img[mask] = torch.tensor([0.0, 0.0, 0.0])
        else:
            vis_img[mask] = torch.tensor(color[:3]).float() # RGB only, ignore alpha if present

    vis_img = vis_img.view(h, w, 3).cpu()

    axs[1, 1].imshow(vis_img)
    axs[1, 1].set_title("Semantic Velocity")
    axs[1, 1].axis("off")
        
    # Augmented semantic velocity map
    b, _, c, h, w = out_sem_vel_aug.shape
    vis_img = torch.zeros((h * w, 3))
    rearranged = rearrange(out_sem_vel_aug[0, t], "c h w -> (h w) c").cpu()
    valid_mask = rearranged[:, 0] != 0
    
    unique_vals, inverse_indices = torch.unique(rearranged, dim=0, return_inverse=True)
    colors = matplotlib.colormaps['tab20'].resampled(unique_vals.shape[0])
    for idx, color in enumerate(colors.colors):
        mask = (inverse_indices == idx)
        if torch.all(unique_vals[idx] == 0):
            vis_img[mask] = torch.tensor([0.0, 0.0, 0.0])
        else:
            vis_img[mask] = torch.tensor(color[:3]).float() # RGB only, ignore alpha if present
        
    vis_img = vis_img.view(h, w, 3).cpu()

    axs[1, 2].imshow(vis_img)
    axs[1, 2].set_title("Augmented Semantic Velocity")
    axs[1, 2].axis("off")
    
    # Semantic class map
    b, _, c, h, w = out_sem_cls.shape
    vis_img = torch.zeros((h * w, 3))
    rearranged = rearrange(out_sem_cls[0, t], "c h w -> (h w) c").cpu()
    valid_mask = rearranged[:, 0] != 0
    
    unique_vals, inverse_indices = torch.unique(rearranged, dim=0, return_inverse=True)
    colors = matplotlib.colormaps['tab20'].resampled(unique_vals.shape[0])
    for idx, color in enumerate(colors.colors):
        mask = (inverse_indices == idx)
        if torch.all(unique_vals[idx] == 0):
            vis_img[mask] = torch.tensor([0.0, 0.0, 0.0])
        else:
            vis_img[mask] = torch.tensor(color[:3]).float() # RGB only, ignore alpha if present

    vis_img = vis_img.view(h, w, 3).cpu()

    axs[1, 3].imshow(vis_img)
    axs[1, 3].set_title("Semantic Class")
    axs[1, 3].axis("off")
    
    # Semantic class map augmented
    b, _, c, h, w = out_sem_cls_aug.shape
    vis_img = torch.zeros((h * w, 3))
    rearranged = rearrange(out_sem_cls_aug[0, t], "c h w -> (h w) c").cpu()
    valid_mask = rearranged[:, 0] != 0
    
    unique_vals, inverse_indices = torch.unique(rearranged, dim=0, return_inverse=True)
    colors = matplotlib.colormaps['tab20'].resampled(unique_vals.shape[0])
    for idx, color in enumerate(colors.colors):
        mask = (inverse_indices == idx)
        if torch.all(unique_vals[idx] == 0):
            vis_img[mask] = torch.tensor([0.0, 0.0, 0.0])
        else:
            vis_img[mask] = torch.tensor(color[:3]).float() # RGB only, ignore alpha if present
    
    vis_img = vis_img.view(h, w, 3).cpu()
    
    axs[1, 4].imshow(vis_img)
    axs[1, 4].set_title("Augmented Semantic Class")
    axs[1, 4].axis("off")
    
    # Clear axs[1, 5]
    axs[1, 5].cla()
    axs[1, 5].axis('off')

    
    # Plot images from different cameras in last row
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    for i in range(6):
        img = imgs[0, t, i].permute(1, 2, 0).cpu().numpy()
        axs[2, i].imshow(img)
        axs[2, i].set_title(cam_names[i])
        axs[2, i].axis('off')

    plt.savefig("augmentation_vis.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)



@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    pruebas(cfg)



if __name__ == "__main__":
    main()
