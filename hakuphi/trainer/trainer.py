import os
from typing import *
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import pytorch_lightning as pl
from warmup_scheduler import GradualWarmupScheduler

from transformers import PreTrainedModel

from hakuphi.utils import instantiate


class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
        *args,
        name="",
        lr: float = 1e-5,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        opt_configs: dict[str, Any] = {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        lr_scheduler: Optional[type[lr_sch.LRScheduler]] = lr_sch.CosineAnnealingLR,
        lr_sch_configs: dict[str, Any] = {
            "T_max": 100_000,
            "eta_min": 1e-7,
        },
        use_warm_up: bool = True,
        warm_up_period: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.train_params: Iterator[nn.Parameter] = None
        self.optimizer = instantiate(optimizer)
        self.opt_configs = opt_configs
        self.lr = lr
        self.lr_sch = instantiate(lr_scheduler)
        self.lr_sch_configs = lr_sch_configs
        self.use_warm_up = use_warm_up
        self.warm_up_period = warm_up_period

    def configure_optimizers(self):
        assert self.train_params is not None
        optimizer = self.optimizer(self.train_params, lr=self.lr, **self.opt_configs)

        lr_sch = None
        if self.lr_sch is not None:
            lr_sch = self.lr_sch(optimizer, **self.lr_sch_configs)

        if self.use_warm_up:
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 1, self.warm_up_period, lr_sch
            )
        else:
            lr_scheduler = lr_sch

        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
            }

    def train(self, mode: bool = True) -> None:
        """Set the model to training mode"""
        for module in self.children():
            module.train(mode)
        opts = self.optimizers()
        if isinstance(opts, list):
            for opt in opts:
                if hasattr(opt, "train"):
                    opt.train(mode)
        else:
            if hasattr(opts, "train"):
                opts.train(mode)

    def eval(self) -> None:
        """Set the model to evaluation mode"""
        for module in self.children():
            module.eval()
        opts = self.optimizers()
        if isinstance(opts, list):
            for opt in opts:
                if hasattr(opt, "eval"):
                    opt.eval()
        else:
            if hasattr(opts, "eval"):
                opts.eval()


class CausalLMTrainer(BaseTrainer):
    def __init__(
        self,
        text_model: PreTrainedModel = nn.Module(),
        lycoris_model: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super(CausalLMTrainer, self).__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["text_model", "lycoris_model"])
        self.text_model = text_model
        self.lycoris_model = lycoris_model
        if lycoris_model is not None:
            self.text_model.eval()
            self.lycoris_model.train()
            self.train_params = chain(
                self.lycoris_model.parameters(),
                (p for p in self.text_model.parameters() if p.requires_grad),
            )
        else:
            self.text_model.train()
            self.train_params = self.text_model.parameters()
        self.epoch = 0
        self.total_loss = 0
        self.total_token_seen = 0
        self.total_token_trained = 0
        self.global_total_token_seen = 0
        self.global_total_token_trained = 0

    def on_train_epoch_end(self) -> None:
        self.epoch += 1
        if self.lycoris_model is not None:
            dir = "./lycoris_weight"
            epoch = self.epoch
            if self._trainer is not None:
                trainer = self._trainer
                epoch = trainer.current_epoch
                if len(trainer.loggers) > 0:
                    if trainer.loggers[0].save_dir is not None:
                        save_dir = trainer.loggers[0].save_dir
                    else:
                        save_dir = trainer.default_root_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = (
                        version if isinstance(version, str) else f"version_{version}"
                    )
                    dir = os.path.join(save_dir, str(name), version, "lycoris_weight")
                else:
                    # if no loggers, use default_root_dir
                    dir = os.path.join(trainer.default_root_dir, "lycoris_weight")
            os.makedirs(dir, exist_ok=True)
            model_weight = {
                k: v for k, v in self.text_model.named_parameters() if v.requires_grad
            }
            lycoris_weight = self.lycoris_model.state_dict() | model_weight
            torch.save(lycoris_weight, os.path.join(dir, f"epoch={epoch}.pt"))

    def training_step(self, batch, idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        token_count = batch["token_count"]
        trained_token_count = batch["trained_token_count"]

        result = self.text_model(
            input_ids=input_ids,
            labels=labels,
        )
        loss = result.loss
        batch_perplexity = torch.exp(loss)

        self.total_loss += loss.detach().item() * trained_token_count
        self.total_token_seen += token_count
        self.total_token_trained += trained_token_count
        current_total_perplexity = torch.exp(
            torch.tensor(self.total_loss / self.total_token_trained)
        )

        if self._trainer is not None:
            total_token_seen = self.total_token_seen
            total_token_trained = self.total_token_trained
            if self.local_rank > 0:
                with open(f"./temp/rank{self.local_rank}.txt", "w") as f:
                    f.write(f"{self.total_token_seen} {self.total_token_trained} {current_total_perplexity} {batch_perplexity}\n")
            else:
                for f in os.listdir("./temp"):
                    with open(os.path.join("./temp", f), "r") as g:
                        line = g.read().strip()
                    line = line.split()
                    if len(line) >= 2:
                        total_token_seen += int(line[0])
                        total_token_trained += int(line[1])
            self.log("train/token_seen", total_token_seen)
            self.log("train/token_trained", total_token_trained)
            self.log("train/batch_perplexity", batch_perplexity)
            self.log("train/total_perplexity", current_total_perplexity)
            self.log("train/loss", loss, on_step=True, logger=True, prog_bar=True)

        return loss
