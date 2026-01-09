import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self, reduction="mean", epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        
        if self.epsilon is not None:
            class_num = x.shape[-1]
            # Label smoothing
            with torch.no_grad():
                if len(label.shape) == 1 or label.shape[-1] != class_num:
                    label = F.one_hot(label, class_num).float()
                label = label * (1 - self.epsilon) + self.epsilon / class_num
            
            log_probs = F.log_softmax(x, dim=-1)
            loss = -(label * log_probs).sum(dim=-1)
            
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            # PyTorch's cross_entropy supports both integer labels and soft labels (probabilities)
            if len(label.shape) == 1:
                label = label.long()
            else:
                label = label.float()
            loss = F.cross_entropy(x, label, reduction=self.reduction)
            
        return {"CELoss": loss}

class CombinedLoss(nn.Module):
    def __init__(self, loss_config):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        for config in loss_config:
            name = list(config.keys())[0]
            weight = config[name].get("weight", 1.0)
            self.loss_weight.append(weight)
            
            # 简单实现，只支持 CELoss
            if name == "CELoss":
                self.loss_func.append(CELoss(**{k:v for k,v in config[name].items() if k != "weight"}))
            else:
                raise NotImplementedError(f"Loss {name} not implemented")

    def forward(self, input, target):
        loss_dict = {}
        combined_loss = 0
        for i, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, target)
            weight = self.loss_weight[i]
            for k, v in loss.items():
                loss_dict[k] = v
                combined_loss += v * weight
        loss_dict["loss"] = combined_loss
        return loss_dict
