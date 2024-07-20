import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from data import dan_prompt, titpop

import torch

torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sch
import torch.utils.data as Data

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from hakuphi.trainer import CausalLMTrainer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


torch.set_default_dtype(torch.bfloat16)
EPOCH = 2
GPUS = 4
BATCH_SIZE = 64
GRAD_ACC = 4
CUT_OFF = 512
LR = 5e-4


def load_tokenizer(
    tokenizer_ref="TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T",
):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_ref)
    titpop.apply_special_tokens(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    config: LlamaConfig,
    tokenizer: LlamaTokenizer,
) -> LlamaForCausalLM:
    config.pad_token_id = tokenizer.eos_token_id
    config.vocab_size = tokenizer.get_vocab().__len__()
    model = LlamaForCausalLM(config)
    return model


def load_trainer(
    model: LlamaForCausalLM, lycoris_model: nn.Module = None, t_max=1000_000
) -> CausalLMTrainer:
    return CausalLMTrainer(
        model,
        lycoris_model,
        name="TITPOP-200M",
        lr=LR,
        optimizer=torch.optim.AdamW,
        opt_configs={
            "weight_decay": 0.01,
            "betas": (0.9, 0.98),
        },
        lr_scheduler=lr_sch.CosineAnnealingLR,
        lr_sch_configs={
            "T_max": t_max,
            "eta_min": 1e-2 * LR,
        },
        use_warm_up=False,
        warm_up_period=1000,
    )


tokenizer: LlamaTokenizer = load_tokenizer()
processor = titpop.processor(
    tokenizer, cutoff_len=CUT_OFF, train_on_inputs=False, padding=True
)


def collate(batch):
    batch = [processor(data) for data in batch]
    attn_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    result = {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": attn_mask,
        "labels": labels,
        "token_count": torch.sum(attn_mask).cpu().item(),
        "trained_token_count": torch.sum(labels != -100).cpu().item(),
    }
    return result


def main():
    # Setup dataset
    dataset1 = titpop.load("danbooru")
    dataset2 = titpop.load("gbc")
    dataset = Data.ConcatDataset([dataset1, dataset2])
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

    config = LlamaConfig(
        vocab_size=32006,
        hidden_size=768,
        intermediate_size=2304,
        num_hidden_layers=20,
        num_attention_heads=768 // 64,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        use_cache=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    text_model = load_model(config, tokenizer)
    text_model.init_weights()
    # text_model = LlamaForCausalLM.from_pretrained("./DanTagGen-delta")
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
        name="TITPOP-200M",
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
        trainer_module,
        train_dataloaders=data_loader,
    )

    text_model.save_pretrained("TITPOP-200M")
    tokenizer.save_pretrained("TITPOP-200M")


if __name__ == "__main__":
    pl.seed_everything(3407)
    main()
