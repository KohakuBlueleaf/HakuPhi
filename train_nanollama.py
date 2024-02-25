import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from data import dan_prompt

import torch

torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sch
import torch.utils.data as Data

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from hakuphi.trainer import CausalLMTrainer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from bitsandbytes import optim as bnb_optim


EPOCH = 1
GPUS = 1
BATCH_SIZE = 1
GRAD_ACC = 2


def load_model(
    config: LlamaConfig,
    tokenizer_ref="TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T",
) -> tuple[LlamaTokenizer, LlamaForCausalLM]:
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(tokenizer_ref)
    tokenizer = dan_prompt.apply_special_tokens(tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.eos_token_id
    config.vocab_size = tokenizer.get_vocab().__len__()
    print(config.vocab_size)
    model = LlamaForCausalLM(config)
    return tokenizer, model


def load_trainer(
    model: LlamaForCausalLM, lycoris_model: nn.Module = None, t_max=1000_000
) -> CausalLMTrainer:
    return CausalLMTrainer(
        model,
        lycoris_model,
        name="NanoLLaMA_promptgen",
        lr=1e-5,
        optimizer=torch.optim.AdamW,
        opt_configs={
            "weight_decay": 0.01,
            "betas": (0.9, 0.98),
        },
        lr_scheduler=lr_sch.CosineAnnealingLR,
        lr_sch_configs={
            "T_max": t_max,
            "eta_min": 1e-2 * 1e-5,
        },
        use_warm_up=False,
        warm_up_period=1000,
    )


def main():
    config = LlamaConfig(
        vocab_size=32006,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=24,
        num_attention_heads=1024 // 64,
        num_key_value_heads=None,
        hidden_act="mish",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    tokenizer, text_model = load_model(config)
    tokenizer: LlamaTokenizer
    text_model: LlamaForCausalLM
    print(sum(param.shape.numel() for param in text_model.parameters()))
    text_model.gradient_checkpointing_enable()

    # Setup dataset
    dataset = dan_prompt.load()
    print(f"Total training step: {len(dataset)*EPOCH//(BATCH_SIZE*GPUS*GRAD_ACC)}")

    def collate(batch):
        batch = [dan_prompt.generate_prompt(data) for data in batch]
        result = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        result["labels"] = result["input_ids"].clone()
        return result

    data_loader = Data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=collate,  # , num_workers=4
    )

    trainer_module = load_trainer(
        text_model,
        None,
        len(dataset) * EPOCH // (BATCH_SIZE * GPUS * GRAD_ACC),
    )
    print(f"Total training step: {len(dataset)*EPOCH//(BATCH_SIZE*GPUS*GRAD_ACC)}")

    # Train!
    logger = None
    logger = WandbLogger(
        name="danbooru_prompt_2M",
        project="NanoLLaMA",
        offline=True,
    )
    trainer = pl.Trainer(
        precision="16-mixed",
        accelerator="gpu",
        devices=GPUS,
        max_epochs=EPOCH,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=GRAD_ACC,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_train_steps=1000),
        ],
        gradient_clip_val=1.0,
        # fast_dev_run=True,
    )
    trainer.fit(
        trainer_module.train(),
        train_dataloaders=data_loader,
    )

    text_model.save_pretrained("nanollama-test")
    tokenizer.save_pretrained("nanollama-test")


def test():
    config = LlamaConfig(
        vocab_size=32006,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=24,
        num_attention_heads=1024 // 64,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        attn_implementation="flash_attention_2",
    )
    tokenizer, text_model = load_model(config)
    tokenizer: LlamaTokenizer
    text_model: LlamaForCausalLM
    print(sum(param.shape.numel() for param in text_model.parameters()))

    # Setup dataset
    dataset = dan_prompt.load()
    print(f"Total training step: {len(dataset)*EPOCH//(BATCH_SIZE*GPUS*GRAD_ACC)}")

    def collate(batch):
        batch = [dan_prompt.generate_prompt(data) for data in batch]
        result = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=384
        )
        result["labels"] = result["input_ids"].clone()
        return result

    data_loader = Data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=collate,  # , num_workers=4
    )
    for x in iter(data_loader):
        print(x["input_ids"].shape)
        print(x["attention_mask"].shape)
        print(x["labels"].shape)
        break


if __name__ == "__main__":
    # test()
    pl.seed_everything(3407)
    main()
