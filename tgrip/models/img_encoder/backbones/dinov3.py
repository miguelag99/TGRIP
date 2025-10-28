import timm
import torch

from collections import OrderedDict
from peft import LoraConfig, TaskType, get_peft_model
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from tgrip.models.img_encoder.backbones.common import Backbone

class DINOv3(Backbone):
    def __init__(
        self,
        version="vit_large_patch16_dinov3.lvd1689m.",
        downsample=16
    ):
        super().__init__()
        self.version = version
        self.downsample = downsample
        
        assert downsample == 16, "Currently only supported for downsample 16"

        self.model = timm.create_model(version, pretrained=True, features_only=True)
        message = f"DINOv3 exists and is loaded at version {version}"
        self._print_loaded_file(message)
        
        # Freeze the backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add LoRA to the model
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=[
                name for name, _ in self.model.named_modules() if "qkv" in name
            ],
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    @rank_zero_only
    def _print_loaded_file(self, message):
        print("# -------- Backbone -------- #")
        print(message, end="\n")

    def forward(self, x, return_all=False):
        endpoints = dict()
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=True):
            res = self.model(x)
        endpoints["map1"] = res[0]
        endpoints["map2"] = res[1]
        endpoints["map3"] = res[2]

        if not return_all:
            list_keys = ["map2", "map3"]
        else:
            list_keys = ["map1", "map2", "map3"]

        return OrderedDict({f"out{i}": endpoints[k] for i, k in enumerate(list_keys)})