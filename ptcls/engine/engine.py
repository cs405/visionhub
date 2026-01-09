import torch
from ..utils import logger
from ..utils.save_load import init_model, save_model
from ..arch import build_model
from ..data import build_dataloader
from ..loss.celoss import CombinedLoss
from ..metric.metrics import CombinedMetrics
from ..optimizer import build_optimizer

class Engine(object):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode

        # 1. Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Build Model
        self.model = build_model(config)
        self.model.to(self.device)

        # 3. Build Loss
        if "Loss" in config:
            self.train_loss_func = CombinedLoss(config["Loss"]["Train"])
            self.eval_loss_func = CombinedLoss(config["Loss"]["Eval"])

        # 4. Build Metric
        if "Metric" in config:
            self.train_metric_func = CombinedMetrics(config["Metric"]["Train"])
            self.eval_metric_func = CombinedMetrics(config["Metric"]["Eval"])

        # 5. Build Dataloader
        if mode == "train":
            self.train_dataloader = build_dataloader(config, "Train", self.device)
        if mode in ["train", "eval"]:
            self.eval_dataloader = build_dataloader(config, "Eval", self.device)

        # 6. Build Optimizer & LR Scheduler
        self.lr_scheduler = None
        if mode == "train":
            epochs = int(self.config["Global"]["epochs"])
            step_each_epoch = len(self.train_dataloader)
            self.optimizer, self.lr_scheduler = build_optimizer(
                config=self.config["Optimizer"],
                model=self.model,
                epochs=epochs,
                step_each_epoch=step_each_epoch,
            )

        # 7. Init Model (Load checkpoints)
        init_model(config, self.model, self.optimizer if mode == "train" else None)

    def train(self):
        epochs = self.config["Global"]["epochs"]
        best_metric = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.train_metric_func.reset()

            for idx, (batch, label) in enumerate(self.train_dataloader):
                batch = batch.to(self.device)
                label = label.to(self.device)

                out = self.model(batch)
                loss_dict = self.train_loss_func(out, label)
                loss = loss_dict["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # lr step（按 iter）
                if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                    self.lr_scheduler.step()

                metrics = self.train_metric_func(out, label)

                if idx % self.config["Global"]["print_batch_step"] == 0:
                    logger.info(
                        f"Epoch [{epoch}/{epochs}], Step [{idx}/{len(self.train_dataloader)}], "
                        f"Loss: {loss.item():.4f}, Top1: {metrics['top1']:.4f}"
                    )

            # lr step（按 epoch）
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # Eval
            eval_metrics = self.eval()
            if eval_metrics["top1"] > best_metric:
                best_metric = eval_metrics["top1"]
                save_model(
                    self.model,
                    self.optimizer,
                    eval_metrics,
                    self.config["Global"]["save_inference_dir"],
                    "best_model",
                )

            save_model(
                self.model,
                self.optimizer,
                eval_metrics,
                self.config["Global"]["save_inference_dir"],
                "latest",
            )

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        self.eval_metric_func.reset()

        for idx, (batch, label) in enumerate(self.eval_dataloader):
            batch = batch.to(self.device)
            label = label.to(self.device)

            out = self.model(batch)
            self.eval_metric_func(out, label)

        metrics = self.eval_metric_func.avg()
        logger.info(f"Eval Results: {metrics}")
        return metrics
