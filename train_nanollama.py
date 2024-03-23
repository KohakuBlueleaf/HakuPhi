import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from data import dan_prompt

import torch

torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sch
import torch.utils.data as Data

from transformers import LlamaForCausalLM, LlamaTokenizer

from hakuphi.trainer import CausalLMTrainer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


EPOCH = 10
GPUS = 4
BATCH_SIZE = 64
GRAD_ACC = 4
CUT_OFF = 384
LR = 2e-4


def load_trainer(
    model: LlamaForCausalLM, lycoris_model: nn.Module = None, t_max=1000_000
) -> CausalLMTrainer:
    return CausalLMTrainer(
        model,
        lycoris_model,
        name="NanoLLaMA_promptgen",
        lr=LR,
        optimizer=torch.optim.AdamW,
        opt_configs={
            "weight_decay": 0.01,
            "betas": (0.9, 0.99),
        },
        lr_scheduler=lr_sch.CosineAnnealingLR,
        lr_sch_configs={
            "T_max": t_max,
            "eta_min": 1e-2 * LR,
        },
        use_warm_up=True,
        warm_up_period=1000,
    )


tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained("./nanollama-test")
processor = dan_prompt.processor(
    tokenizer, cutoff_len=CUT_OFF, train_on_inputs=False, padding=True
)


def collate(batch):
    batch = [processor(data) for data in batch]
    result = {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack(
            [x["attention_mask"] for x in batch]
        ),
        "labels": torch.stack([x["labels"] for x in batch]),
    }
    return result


def main():
    # Setup dataset
    dataset = dan_prompt.load()
    print(f"Total training step: {len(dataset)*EPOCH//(BATCH_SIZE*GPUS*GRAD_ACC)}")

    data_loader = Data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=collate,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # text_model = load_model(config, tokenizer)
    text_model = LlamaForCausalLM.from_pretrained("./nanollama-ft")
    text_model.init_weights()
    print(sum(param.shape.numel() for param in text_model.parameters()))
    text_model.gradient_checkpointing_enable()

    trainer_module = load_trainer(
        text_model.to(torch.float),
        None,
        len(dataset) * EPOCH // (BATCH_SIZE * GPUS * GRAD_ACC),
    )
    print(f"Total training step: {len(dataset)*EPOCH//(BATCH_SIZE*GPUS*GRAD_ACC)}")

    # Train!
    logger = None
    logger = WandbLogger(
        name="NanoLLaMA-400M_dan-5.4M-pretrain",
        project="NanoLLaMA",
        # offline=True,
    )
    trainer = pl.Trainer(
        precision="bf16-mixed",
        accelerator="gpu",
        devices=GPUS,
        max_epochs=EPOCH,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=GRAD_ACC,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_train_steps=1000),
            ModelCheckpoint(every_n_epochs=1),
        ],
        gradient_clip_val=1.0,
        # fast_dev_run=True,
    )
    trainer.fit(
        trainer_module.train(),
        train_dataloaders=data_loader,
    )

    text_model.save_pretrained("nanollama-pretrain2")
    tokenizer.save_pretrained("nanollama-pretrain2")


if __name__ == "__main__":
    pl.seed_everything(3407)
    main()
