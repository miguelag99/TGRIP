import carla
import torch
import hydra

from pytorch_lightning import LightningDataModule

from tgrip import utils

log = utils.get_pylogger(__name__)

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load a different map
client.load_world('Town05')

print("Connected to CARLA and loaded Town05")

# Prediction model

with hydra.initialize(version_base="1.3", config_path="./configs"):
    predictor_cfg = hydra.compose(config_name="val")
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

log.info(f"Instantiating model <{predictor_cfg.model._target_}>")
model = hydra.utils.instantiate(predictor_cfg.model).to(device)

ckpt = torch.load(predictor_cfg.ckpt.path, map_location=device, weights_only=False)

model = utils.load_state_model(
    model,
    ckpt,
    keys_to_freeze="all",
    # Remove semantic head 'net.semantic_head'
    keys_to_load=['net.backbone', 'net.neck', 'net.view_transform',
                    'net.decoder', 'net.temporal', 'net.heads',
                    'net.coord_selector', 'net.query_gen'],
    verbose=1,
)

log.info(f"Instantiating datamodule <{predictor_cfg.data._target_}>")
datamodule: LightningDataModule = hydra.utils.instantiate(predictor_cfg.data)
datamodule.setup()
dataset = datamodule.val_dataloader()

x = next(iter(dataset))

for k, v in x.items():
    if isinstance(v, torch.Tensor):
        x[k] = v.to(device)

with torch.inference_mode():
    model.eval()
    out = model.net(**x)

import pdb; pdb.set_trace()

