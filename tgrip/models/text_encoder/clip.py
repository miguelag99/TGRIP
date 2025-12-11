import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze=True):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze:
            print("Freezing CLIP parameters.")
            for p in self.clip_model.parameters():
                p.requires_grad = False
        self.out_dim = self._get_output_dim()

    def _get_output_dim(self):
        return self.clip_model.text_projection.out_features

    def forward(self, texts):
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.clip_model.device)
        
        outputs = self.clip_model.get_text_features(**inputs)
        
        return outputs
    
    def get_visual_features(self, images):
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            do_center_crop=False,
            do_resize=True,
        ).to(self.clip_model.device)
        
        outputs = self.clip_model.get_image_features(**inputs)
        
        return outputs
