import torch
from transformers import CLIPTextModel, CLIPTokenizer

class CLIPEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.out_dim = self._get_output_dim()

    def _get_output_dim(self):
        return self.encoder.config.hidden_size

    def forward(self, texts):
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.encoder.device)
        out = self.encoder(**tokens)
        return out.pooler_output