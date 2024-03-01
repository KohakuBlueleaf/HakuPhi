import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from time import time_ns

from data import dan_prompt

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm

from hakuphi.trainer import CausalLMTrainer
from hakuphi.inference import generate

import pytorch_lightning as pl


def load_tokenizer(tokenizer_ref="TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_ref)
    dan_prompt.apply_special_tokens(tokenizer)
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


tokenizer: LlamaTokenizer = load_tokenizer()


@torch.no_grad()
def main():
    config = LlamaConfig(
        vocab_size=32006,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=24,
        num_attention_heads=1024 // 64,
        hidden_act="mish",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        use_cache=True,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    text_model = load_model(config, tokenizer)
    full_module = CausalLMTrainer.load_from_checkpoint(
        "./NanoLLaMA/epoch=5-step=45000.ckpt",
        text_model=text_model,
    )
    text_model = full_module.text_model.float().cpu()

    generate(
        model=text_model,
        tokenizer=tokenizer,
        prompt="rating: sensitive",
        max_new_tokens=128,
    )

    prev = ""
    prompt = f"""
rating: sensitive
artist: <|empty|>
characters: <|empty|>
copyrights: <|empty|>
target: <|long|>
general: 1girl, solo, dragon girl, dragon tail, blue hair, closed mouth, purple eyes<|input_end|>
""".lstrip()
    t0 = time_ns()
    for i in tqdm(
        generate(
            model=text_model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=1.2,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.00,
            max_new_tokens=256,
            stream_output=True,
        ),
        disable=True,
    ):
        if len(i) > len(prev):
            new_len = len(i) - len(prev)
            print(i[len(prev) :], end="", flush=True)
            prev = i
        pass
    t1 = time_ns()
    result = i[len(prompt) :]
    result_tokens = len(tokenizer.tokenize(result))
    print()
    print("=" * 50)
    print(f"Total generated tokens: {result_tokens}")
    print(f"Total cost time: {(t1-t0)/1e9:.2f}s")
    print(f"Average Speed: {(result_tokens/((t1-t0)/1e9)):.2f} tokens/sec")
    print()


if __name__ == "__main__":
    main()
