import torch
from transformers import SiglipModel, SiglipProcessor

class SiglipEncoder(torch.nn.Module):    
    def __init__(self, model_name="google/siglip2-base-patch16-224", freeze=True):
        super().__init__()
        self.siglip_model = SiglipModel.from_pretrained(model_name)
        self.processor = SiglipProcessor.from_pretrained(model_name)
        
        if freeze:
            print("Freezing SigLIP parameters.")
            for p in self.siglip_model.parameters():
                p.requires_grad = False
        self.out_dim = self._get_output_dim()

    def _get_output_dim(self):
        return self.siglip_model.text_model.head.out_features
    
    def forward(self, texts):
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.siglip_model.device)
        
        outputs = self.siglip_model.get_text_features(**inputs)
        
        return outputs
    
    def get_visual_features(self, images):
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            do_center_crop=False,
            do_resize=True,
        ).to(self.siglip_model.device)
        
        outputs = self.siglip_model.get_image_features(**inputs)
        
        return outputs