import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from data import guanaco, final

import torch

torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import torch.utils.data as Data

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from hakuphi.model import PhiForCausalLM
from hakuphi.trainer import CausalLMTrainer
from hakuphi.tools import add_tokens
from hakuphi.attn_patcher import apply_attn_algo

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from bitsandbytes import optim as bnb_optim
from prodigyopt import Prodigy
from train_utils import ProdigyLRMonitor


EPOCH = 5
GPUS = 2
BATCH_SIZE = 16
GRAD_ACC = 2


def load_model(
    path="microsoft/phi-2", load_extra_tokens=True
) -> tuple[PreTrainedTokenizer, PhiForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = PhiForCausalLM.from_pretrained(path)
    if load_extra_tokens:
        extra_tokens = add_tokens.load_extra_tokens()
        tokenizer.add_tokens(extra_tokens)
        model.resize_token_embeddings(len(tokenizer))
    apply_attn_algo(model, "xformers")
    return tokenizer, model


def load_trainer(
    model: PreTrainedModel, lycoris_model: nn.Module = None, t_max=1000_000
) -> CausalLMTrainer:
    return CausalLMTrainer(
        model,
        lycoris_model,
        name="Phi-MultiLingual",
        lr=0.5,
        optimizer=Prodigy,
        opt_configs={
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "use_bias_correction": True,
            "decouple": True,
        },
        lr_scheduler=lr_sch.CosineAnnealingLR,
        lr_sch_configs={
            "T_max": t_max,
            "eta_min": 1e-2,
        },
        use_warm_up=False,
        warm_up_period=1000,
    )


def load_guanaco_dataset(tokenizer):
    raw_datas = guanaco.load("mini")["train"]
    processor = guanaco.processor(
        tokenizer, cutoff_len=1024, train_on_inputs=False, padding=True
    )
    dataset = raw_datas.shuffle().map(processor, desc="load data", batch_size=320)
    return dataset


def load_final_dataset(tokenizer):
    raw_datas = final.load("all")["train"]
    processor = final.processor(
        tokenizer, cutoff_len=1024, train_on_inputs=False, padding=True
    )
    dataset = raw_datas.shuffle().map(processor, desc="load data", batch_size=320)
    return dataset


def lycoris_wrapper(
    main_module: nn.Module,
    lycoris_settings: dict,
    lycoris_presets: dict = None,
):
    from lycoris.wrapper import create_lycoris, LycorisNetwork

    lycoris_settings = lycoris_settings
    if lycoris_presets is not None:
        LycorisNetwork.apply_preset(lycoris_presets)
    lycoris_net = create_lycoris(module=main_module, **lycoris_settings)
    lycoris_net.apply_to()
    return lycoris_net


def main():
    # Loading models and datasets
    tokenizer, text_model = load_model(load_extra_tokens=False)

    # Setup phi model's extra configs
    text_model.half()
    text_model.gradient_checkpointing_enable()
    text_model.use_neftune = True
    text_model.neft_alpha = 50
    apply_attn_algo(text_model, "xformers")

    # FP8
    # text_model.transformer.h.to(torch.float8_e4m3fn)
    # text_model.lm_head.to(torch.float8_e4m3fn)
    text_model.requires_grad_(False)

    # wrap lycoris
    lycoris_model = None
    lycoris_settings = {
        "multiplier": 1.0,
        "linear_dim": 100000,
        "linear_alpha": 0,
        "factor": 16,
        "algo": "lokr",
    }
    lycoris_presets = {"target_module": ["ParallelBlock"]}
    lycoris_model = lycoris_wrapper(text_model, lycoris_settings, lycoris_presets)

    # Setup dataset
    main_dataset = load_final_dataset(tokenizer)
    reg_dataset = load_guanaco_dataset(tokenizer)
    dataset = Data.ConcatDataset([reg_dataset, main_dataset])

    trainer_module = load_trainer(
        text_model,
        lycoris_model,
        len(dataset) * EPOCH // (BATCH_SIZE * GPUS * GRAD_ACC),
    )
    print(f"Total training step: {len(dataset)*EPOCH//(BATCH_SIZE*GPUS*GRAD_ACC)}")

    def collate(batch):
        return {
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
            "attention_mask": torch.stack(
                [torch.tensor(x["attention_mask"]) for x in batch]
            ),
            "labels": torch.stack([torch.tensor(x["labels"]) for x in batch]),
        }

    data_loader = Data.DataLoader(
        dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=4
    )

    # Train!
    logger = None
    logger = WandbLogger(
        name="phi-test",
        project="Haku-Phi",
        # offline = True,
    )
    trainer = pl.Trainer(
        precision="16-mixed",
        accelerator="gpu",
        devices=GPUS,
        max_epochs=EPOCH,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        callbacks=[
            ProdigyLRMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_train_steps=1000),
        ],
        gradient_clip_val=1.0,
        # fast_dev_run=True
    )
    trainer.fit(
        trainer_module.train(),
        train_dataloaders=data_loader,
    )

    # Test?
    model_weight = {k: v for k, v in text_model.named_parameters() if v.requires_grad}
    lycoris_weight = lycoris_model.state_dict() | model_weight
    torch.save(lycoris_weight, "lycoris_weight_final.pt")


if __name__ == "__main__":
    pl.seed_everything(3407)
    main()
