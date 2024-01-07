from typing import *
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from einops import rearrange, repeat

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from .utils.xformers_utils import XFORMERS_AVAIL, memory_efficient_attention
from .model.modeling_phi import PhiForCausalLM, SelfAttention, CrossAttention


def vanilla_attention(
    q,
    k,
    v,
    causal_mask=False,
    padding_mask=None,
    softmax_scale=1.0,
    dropout=nn.Dropout(),
):
    mask = 0
    if causal_mask is not None:
        mask = mask + causal_mask[..., : k.size(1)].to(dtype=q.dtype)
    if padding_mask is not None:
        mask = mask + padding_mask[..., : k.size(1)].to(dtype=q.dtype)
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    scores = scores + mask
    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    attention = dropout(attention)
    return torch.einsum("bhts,bshd->bthd", attention, v)


def fast_attn_wrapper(fast_attn, extra_rearrange=None):
    if extra_rearrange is not None:
        pre_re = "->".join(extra_rearrange)
        post_re = "->".join(extra_rearrange[::-1])

    def runner(
        q,
        k,
        v,
        causal_mask=False,
        padding_mask=None,
        softmax_scale=1.0,
        dropout=nn.Dropout(),
    ):
        batch_size, seq_len, head_nums, head_dim = q.shape
        seq_len_k = k.shape[1]
        drop_rate = dropout.p if isinstance(dropout, nn.Module) else dropout

        mask = 0
        if causal_mask is not None:
            causal_mask = repeat(
                causal_mask, "t s -> b h t s", b=batch_size, h=head_nums
            )
            mask = mask + causal_mask.to(dtype=q.dtype)
        if padding_mask is not None:
            padding_mask = repeat(
                padding_mask, "b 1 1 m -> b h t m", h=head_nums, t=seq_len
            )
            mask = mask + padding_mask.to(dtype=q.dtype)
        if not isinstance(mask, torch.Tensor) and mask == 0:
            mask = None
        else:
            mask = mask[..., :seq_len_k]

        if extra_rearrange:
            q = rearrange(q, pre_re)
            k = rearrange(k, pre_re)
            v = rearrange(v, pre_re)
        output = fast_attn(q, k, v, mask, drop_rate, scale=softmax_scale)
        if extra_rearrange:
            output = rearrange(output, post_re)
        return output

    return runner


ATTN_ALGO = {
    "vanilla": vanilla_attention,
    "torch-sdp": fast_attn_wrapper(
        getattr(F, "scaled_dot_product_attention", None), ("b t h s", "b h t s")
    ),
    "xformers": fast_attn_wrapper(memory_efficient_attention),
}


@dataclasses.dataclass
class AttnConfig:
    causal: bool = True
    softmax_scale: Optional[float] = None
    drop: nn.Dropout = None
    attn_algo: str = "vanilla"


def attn_patcher(func):
    def attn_module_wrapper(
        attn_module: SelfAttention | CrossAttention, algo="vanilla"
    ):
        causal = attn_module.causal
        softmax_scale = attn_module.softmax_scale
        drop = attn_module.drop
        config = AttnConfig(
            causal=causal, softmax_scale=softmax_scale, drop=drop, attn_algo=algo
        )

        def runner(*args, **kwargs):
            return func(config, *args, **kwargs)

        return runner

    return attn_module_wrapper


@attn_patcher
def sforward(
    self: AttnConfig,
    qkv: torch.FloatTensor,
    causal: bool = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    **kwargs,
):
    causal = self.causal if causal is None else causal
    batch_size, seq_len, *_ = qkv.shape
    q, k, v = qkv.unbind(dim=2)

    padding_mask = causal_mask = None
    if attention_mask is not None:
        padding_mask = torch.full(
            (batch_size, math.ceil(seq_len / 8) * 8), float("-inf"), device=q.device
        )
        padding_mask.masked_fill_(attention_mask, 0.0)
        padding_mask = rearrange(padding_mask, "b s -> b 1 1 s")
    if causal:
        causal_mask = torch.triu(
            torch.full(
                (seq_len, math.ceil(seq_len / 8) * 8), float("-inf"), device=q.device
            ),
            1,
        )
    softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

    attn_func = ATTN_ALGO[getattr(self, "attn_algo", "vanilla")]
    output = attn_func(q, k, v, causal_mask, padding_mask, softmax_scale, self.drop)

    return output


@attn_patcher
def xforward(
    self: AttnConfig,
    q: torch.FloatTensor,
    kv: torch.FloatTensor,
    causal: bool = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    **kwargs,
):
    causal = self.causal if causal is None else causal
    batch_size, seq_len_q, *_ = q.shape
    seq_len_k = kv.shape[1]
    assert (
        kv.shape[0] == batch_size
        and kv.shape[3] == q.shape[2]
        and kv.shape[4] == q.shape[3]
    )

    k, v = kv.unbind(dim=2)

    padding_mask = causal_mask = None
    if attention_mask is not None:
        padding_mask = torch.full(
            (batch_size, math.ceil(seq_len_k / 8) * 8), float("-inf"), device=q.device
        )
        padding_mask[:seq_len_k].masked_fill_(attention_mask, 0.0)
        padding_mask = rearrange(padding_mask, "b s -> b 1 1 s")
    if causal:
        mask = torch.triu(
            torch.full((seq_len_q, seq_len_q), float("-inf"), device=q.device), 1
        )
        align8 = math.ceil(seq_len_k / 8) * 8
        causal_mask = F.pad(
            mask, (seq_len_k - seq_len_q, align8 - seq_len_k), value=0.0
        )
    softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

    attn_func = ATTN_ALGO[getattr(self, "attn_algo", "vanilla")]
    output = attn_func(q, k, v, causal_mask, padding_mask, softmax_scale, self.drop)

    return output


def apply_attn_algo(model: PhiForCausalLM, algo="vanilla"):
    if algo == "torch-sdp":
        assert torch.__version__ >= "2.0.0"
    elif algo == "xformers":
        assert XFORMERS_AVAIL
    elif algo != "vanilla":
        print(f"unknown attn algo: {algo}. Using 'vanilla' instead.")
        algo = "vanilla"

    for module in model.modules():
        if module.__class__.__name__.endswith("Attention"):
            if module.__class__.__name__ == "SelfAttention":
                module.forward = sforward(module, algo=algo)
            elif module.__class__.__name__ == "CrossAttention":
                module.forward = xforward(module, algo=algo)
            else:
                continue


if __name__ == "__main__":
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        model = PhiForCausalLM.from_pretrained(
            "microsoft/phi-1_5", trust_remote_code=True
        )
        model = model.half().cuda()

        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-1_5", trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        input = tokenizer(
            ["Python is"],
            return_tensors="pt",
        )
        # Use original implementation to ensure our implementation is correct
        result1 = model.generate(
            input["input_ids"].cuda(),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=32,
        )
        result1 = tokenizer.decode(result1.sequences[0])
        apply_attn_algo(model, algo="torch-sdp")
        result2 = model.generate(
            input["input_ids"].cuda(),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=32,
        )
        result2 = tokenizer.decode(result2.sequences[0])
        apply_attn_algo(model, algo="xformers")
        result3 = model.generate(
            input["input_ids"].cuda(),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=32,
        )
        result3 = tokenizer.decode(result3.sequences[0])
        print(result1)
        print(result2)
        print(result3)

        input = tokenizer(
            # Some different sentences with varius lengths
            [
                "The sun shines brightly in the clear blue sky.",
                "I'm studying computer science at National Tsing Hua University in Taiwan.",
                "Python is a versatile programming language that's used for a wide range of applications.",
                "Music has the power to evoke strong emotions and memories.",
                "In my free time, I enjoy reading literature and exploring visual arts.",
            ],
            return_tensors="pt",
            padding=True,
        )
        embeddings = model.get_input_embeddings()(input["input_ids"].cuda())
        mask = input["attention_mask"].cuda()

        apply_attn_algo(model, algo="vanilla")
        result1 = model.layers[1](embeddings, attention_mask=mask.bool())
        apply_attn_algo(model, algo="torch-sdp")
        result2 = model.layers[1](embeddings, attention_mask=mask.bool())
        apply_attn_algo(model, algo="xformers")
        result3 = model.layers[1](embeddings, attention_mask=mask.bool())
        print(
            F.mse_loss(result1, result2),
            F.mse_loss(result1, result3),
            F.mse_loss(result2, result3),
        )
