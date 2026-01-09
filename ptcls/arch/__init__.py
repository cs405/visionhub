import torch.nn as nn
from .backbone import build_backbone
from .head import EmbeddingHead

__all__ = [
    "BaseModel",
    "build_model",
]


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config["Backbone"])

        # optional embedding head
        self.head = None
        head_cfg = config.get("Head") if isinstance(config, dict) else None
        if head_cfg and head_cfg.get("name") == "EmbeddingHead":
            # 优先用 backbone.feature_dim（正规 feature 维度）
            in_dim = getattr(self.backbone, "feature_dim", None)
            if in_dim is None:
                # fallback（极端情况）：仍用 class_num
                in_dim = int(config["Backbone"].get("class_num", 1000))
            in_dim = int(in_dim)

            emb_dim = int(head_cfg.get("embedding_size", 512))
            with_relu = bool(head_cfg.get("with_relu", False))
            with_l2norm = bool(head_cfg.get("with_l2norm", True))
            bn_affine = bool(head_cfg.get("bn_affine", True))
            self.head = EmbeddingHead(
                in_dim=in_dim,
                embedding_size=emb_dim,
                with_relu=with_relu,
                with_l2norm=with_l2norm,
                bn_affine=bn_affine,
            )

    def forward(self, x):
        # Prefer pooled feature for embedding; logits are only auxiliary.
        feat = None
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)

        logits = self.backbone.fc(feat) if feat is not None and hasattr(self.backbone, "fc") else self.backbone(x)
        out = {"logits": logits}
        if feat is not None:
            out["feature"] = feat

        if self.head is not None:
            if feat is None:
                out["embedding"] = self.head(logits)
            else:
                out["embedding"] = self.head(feat)
        return out


def build_model(config):
    return BaseModel(config["Arch"])
