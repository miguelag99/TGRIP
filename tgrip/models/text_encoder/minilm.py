import torch
from transformers import AutoTokenizer, AutoModel

class MiniLMEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        freeze=True
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.out_dim = self._get_output_dim()

    def _get_output_dim(self):
        return self.encoder.pooler.dense.out_features

    def forward(self, texts):
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.encoder.device)
        out = self.encoder(**tokens)
        sentence_embeddings = mean_pooling(out, tokens['attention_mask'])
        return sentence_embeddings


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
