import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableProjector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, embeddings):
        return self.projector(embeddings)


class InBatchContrastiveRetrievalLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb, positive_emb):
        query_emb = F.normalize(query_emb, dim=-1)
        positive_emb = F.normalize(positive_emb, dim=-1)
        logits = query_emb @ positive_emb.t()
        logits = logits / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        query_to_positive = F.cross_entropy(logits, labels)
        positive_to_query = F.cross_entropy(logits.t(), labels)
        return 0.5 * (query_to_positive + positive_to_query)


def save_projector_checkpoint(path, model, args=None):
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "output_dim": model.output_dim,
        },
    }
    if args is not None:
        payload["args"] = vars(args) if hasattr(args, "__dict__") else args
    torch.save(payload, path)


def load_projector_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        model = LearnableProjector(
            input_dim=config.get("input_dim", 768),
            hidden_dim=config.get("hidden_dim", 512),
            output_dim=config.get("output_dim", 256),
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = LearnableProjector()
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
