"""XBM Contrastive Loss and DKD Loss

XBM对比损失和解耦知识蒸馏损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss_XBM(nn.Module):
    """Cross-Batch Memory (XBM) Contrastive Loss

    使用跨批次内存的对比损失
    Paper: Cross-Batch Memory for Embedding Learning

    Args:
        memory_size: 内存库大小
        feat_dim: 特征维度
        temperature: 温度参数
    """

    def __init__(self, memory_size=8192, feat_dim=128, temperature=0.07):
        super().__init__()
        self.memory_size = memory_size
        self.feat_dim = feat_dim
        self.temperature = temperature

        # 内存队列
        self.register_buffer('memory_queue', torch.randn(memory_size, feat_dim))
        self.register_buffer('memory_labels', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # 归一化内存
        self.memory_queue = F.normalize(self.memory_queue, p=2, dim=1)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, features, labels):
        """更新内存队列"""
        batch_size = features.size(0)

        ptr = int(self.queue_ptr)

        # 循环队列
        if ptr + batch_size > self.memory_size:
            # 分两次填充
            remaining = self.memory_size - ptr
            self.memory_queue[ptr:] = features[:remaining]
            self.memory_labels[ptr:] = labels[:remaining]

            self.memory_queue[:batch_size - remaining] = features[remaining:]
            self.memory_labels[:batch_size - remaining] = labels[remaining:]

            ptr = batch_size - remaining
        else:
            self.memory_queue[ptr:ptr + batch_size] = features
            self.memory_labels[ptr:ptr + batch_size] = labels
            ptr = (ptr + batch_size) % self.memory_size

        self.queue_ptr[0] = ptr

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) 特征
            labels: (B,) 标签
        """
        # 归一化
        features = F.normalize(features, p=2, dim=1)

        # 当前batch的相似度
        sim_batch = torch.matmul(features, features.T) / self.temperature

        # 与内存的相似度
        sim_memory = torch.matmul(features, self.memory_queue.T) / self.temperature

        # 合并相似度矩阵
        sim_matrix = torch.cat([sim_batch, sim_memory], dim=1)

        # 构建标签mask
        labels_batch = labels.unsqueeze(1)
        mask_batch = (labels_batch == labels_batch.T).float()
        mask_batch.fill_diagonal_(0)

        labels_memory = self.memory_labels.unsqueeze(0)
        mask_memory = (labels_batch == labels_memory).float()

        mask = torch.cat([mask_batch, mask_memory], dim=1)

        # 对比损失
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # 平均正样本log概率
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos.mean()

        # 更新内存
        self._dequeue_and_enqueue(features.detach(), labels)

        return loss


class DistillationDKDLoss(nn.Module):
    """Decoupled Knowledge Distillation Loss

    解耦知识蒸馏损失
    Paper: Decoupled Knowledge Distillation

    Args:
        temperature: 温度参数
        alpha: 目标类知识权重
        beta: 非目标类知识权重
    """

    def __init__(self, temperature=4.0, alpha=1.0, beta=8.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: (B, C) 学生logits
            teacher_logits: (B, C) 教师logits
            labels: (B,) 标签
        """
        T = self.temperature

        # Softmax
        student_prob = F.softmax(student_logits / T, dim=1)
        teacher_prob = F.softmax(teacher_logits / T, dim=1)

        # 目标类知识 (TCKD)
        batch_size = student_logits.size(0)
        target_mask = F.one_hot(labels, num_classes=student_logits.size(1)).bool()

        student_target = student_prob[target_mask]
        teacher_target = teacher_prob[target_mask]

        tckd_loss = F.binary_cross_entropy(
            student_target,
            teacher_target,
            reduction='mean'
        )

        # 非目标类知识 (NCKD)
        student_non_target = student_prob.masked_fill(target_mask, 0)
        teacher_non_target = teacher_prob.masked_fill(target_mask, 0)

        # 归一化非目标概率
        student_non_target = student_non_target / (student_non_target.sum(dim=1, keepdim=True) + 1e-8)
        teacher_non_target = teacher_non_target / (teacher_non_target.sum(dim=1, keepdim=True) + 1e-8)

        nckd_loss = F.kl_div(
            torch.log(student_non_target + 1e-8),
            teacher_non_target,
            reduction='batchmean'
        )

        # 总损失
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        loss = loss * (T * T)

        return loss


__all__ = [
    'ContrastiveLoss_XBM',
    'DistillationDKDLoss',
]

