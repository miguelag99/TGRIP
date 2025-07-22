import torch
from transformers import AutoTokenizer, AutoModel

class BertMiniEncoder(torch.nn.Module):
    def __init__(self, pooling="mean", freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
        self.encoder = AutoModel.from_pretrained("prajjwal1/bert-mini")
        self.pooling = pooling
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

        if self.pooling == "cls":
            return out.last_hidden_state[:, 0, :]
        else:
            mask = tokens["attention_mask"].unsqueeze(-1)
            x = out.last_hidden_state * mask
            return x.sum(dim=1) / mask.sum(dim=1)