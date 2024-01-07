from typing import *

from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


class ProdigyLRMonitor(LearningRateMonitor):
    def _get_optimizer_stats(
        self, optimizer: Optimizer, names: List[str]
    ) -> Dict[str, float]:
        stats = {}
        param_groups = optimizer.param_groups
        use_betas = "betas" in optimizer.defaults

        for pg, name in zip(param_groups, names):
            lr = self._extract_lr(pg, name)
            stats.update(lr)
            momentum = self._extract_momentum(
                param_group=pg,
                name=name.replace(name, f"{name}-momentum"),
                use_betas=use_betas,
            )
            stats.update(momentum)
            weight_decay = self._extract_weight_decay(pg, f"{name}-weight_decay")
            stats.update(weight_decay)

        return stats

    def _extract_lr(self, param_group: Dict[str, Any], name: str) -> Dict[str, Any]:
        lr = param_group["lr"]
        d = param_group.get("d", 1)
        self.lrs[name].append(lr * d)
        return {name: lr * d}
