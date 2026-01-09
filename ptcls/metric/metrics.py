import torch
import torch.nn as nn

class TopkAcc(nn.Module):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk
        self.reset()

    def reset(self):
        self.corrects = {k: 0 for k in self.topk}
        self.total = 0

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        
        metric_dict = {}
        maxk = max(self.topk)
        batch_size = label.size(0)

        _, pred = x.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.reshape(1, -1).expand_as(pred))

        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            self.corrects[k] += correct_k.item()
            metric_dict[f"top{k}"] = correct_k.item() / batch_size
        
        self.total += batch_size
        return metric_dict

    def avg(self):
        return {f"top{k}": self.corrects[k] / self.total for k in self.topk}

class CombinedMetrics(nn.Module):
    def __init__(self, metric_config):
        super().__init__()
        self.metrics = []
        for config in metric_config:
            name = list(config.keys())[0]
            params = config[name]
            if params is None:
                params = {}
            if name == "TopkAcc":
                self.metrics.append(TopkAcc(**params))
            else:
                raise NotImplementedError(f"Metric {name} not implemented")

    def forward(self, input, target):
        metric_dict = {}
        for metric in self.metrics:
            m = metric(input, target)
            metric_dict.update(m)
        return metric_dict

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def avg(self):
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric.avg())
        return metric_dict
