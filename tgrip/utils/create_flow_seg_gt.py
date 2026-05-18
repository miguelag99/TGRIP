import hydra
import pyrootutils
import os

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from typing import Optional
from tqdm import tqdm
from safetensors.torch import save_file
from concurrent.futures import ThreadPoolExecutor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tgrip import utils

log = utils.get_pylogger(__name__)
save_executor = ThreadPoolExecutor(max_workers=1)

def async_save(tensors, filename):
    """Background task to save file"""
    try:
        save_file(tensors, filename)
    except Exception as e:
        print(f"Failed to save {filename}: {e}")
        
def create_flow_seg_gt(
    cfg: DictConfig,
    save_dir: str = "flow_seg_gt",
    cam_T_P: list = [[-2,0],[-1,0],[0,0]],
    bev_T_P: list = [[0,0]], 
):
    """
    This function iterates through the trainval split of nuScenes and saves the
    flow and segmentation ground truth for each sample.
    
    By default the dataloader uses multiple past frames and returns multiple
    future frames, using the present frame as the reference.
    You can change this behavior setting:
        - cfg.data.cam_T_P (past): [[-2,0],[-1,0],[0,0]] by default, 3 past frames 
            referencing the present frame
        - cfg.data.bev_T_P (output): [[0,0]] by default.
        
    Args:
        cfg (DictConfig): Hydra config object.
        save_dir (str): Directory to save the generated GT.
        cam_T_P (list): List of [t, P] pairs for camera input frames, where t is the time offset (negative for past, 0 for present) and P is
            the corresponding position in the output sequence. Default is [[-2,0],[-1,0],[0,0]] for 3 past frames referencing the present frame.
        bev_T_P (list): List of [t, P] pairs for BEV output frames, where t is the time offset (positive for future) and P is the corresponding position in
            the output sequence. Default is [[0,0]] for a single future frame referencing the present frame.
    """
    
    cfg.data.prefetch_factor = None
    cfg.data.num_workers = 0
    cfg.data.batch_size = 1 # Fix batch size to 1 to save each sample separately
    
    # Remove BEV data augmentations
    cfg.data.coeffs.bev_aug.trans_rot =  [0.,0.,0.,0.,0.,0.]
    
    # Objects filtering
    cfg.data.img_params.min_visibility = 1  # 1 [0-40%], 2 [40-60%], 3 [60-80%] and 4 [80-100%]
    cfg.data.filters_cat = ['vehicle', 'pedestrian']
    
    cfg.data.version = 'mini'
    cfg.data.train_shuffle = False
    
    # Desired info
    cfg.data.cam_T_P = cam_T_P
    cfg.data.bev_T_P = bev_T_P
    t_index = 0 # T inde to save sample. If you change bev_T_P, change this index to save the desired sample.
    cfg.data.keep_input_detection = False
    cfg.data.keep_input_binimg = True   # Segmentation GT
    cfg.data.keep_input_centr_offs = False  # Centerness and offsets GT
    cfg.data.keep_input_sampling = False
    cfg.data.keep_input_flow_map = True # Flow GT
    cfg.data.keep_input_semantic_maps = False
    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate train samples
    dataset = datamodule.train_dataloader()
    
    for batch in tqdm(dataset, desc="Generating train GT"):
        
        batch_data = {}
        
        sample_token = batch['sample_tokens'][0][t_index]['token']
        segmentation = batch['binimg'][0,t_index,0].cpu()
        flow = batch['flow_map'][0,t_index].cpu()
        batch_data['segmentation'] = segmentation
        batch_data['flow'] = flow
        
        save_path = os.path.join(save_dir, f"{sample_token}.safetensors")
        save_executor.submit(async_save, batch_data, save_path)
    
    # Generate val samples
    dataset = datamodule.val_dataloader()
    for batch in tqdm(dataset, desc="Generating val GT"):
        
        batch_data = {}
        
        sample_token = batch['sample_tokens'][0][t_index]['token']
        segmentation = batch['binimg'][0,t_index,0].cpu()
        flow = batch['flow_map'][0,t_index].cpu()
        batch_data['segmentation'] = segmentation
        batch_data['flow'] = flow
        
        save_path = os.path.join(save_dir, f"{sample_token}.safetensors")
        save_executor.submit(async_save, batch_data, save_path)
        

@hydra.main(version_base="1.3", config_path="../../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.modif_config_based_on_flags(cfg)
    create_flow_seg_gt(cfg)

if __name__ == "__main__":
    main()