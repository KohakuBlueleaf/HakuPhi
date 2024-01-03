from time import time_ns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from tqdm import tqdm

from hakuphi.model import PhiForCausalLM
from hakuphi.attn_patcher import apply_attn_algo
from hakuphi.inference import generate

from lycoris.wrapper import create_lycoris_from_weights


with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    model: PhiForCausalLM = PhiForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    model = model.half()
    
    lycoris_sd = torch.load('Haku-Phi/version_None/lycoris_weight/epoch=3.pt')
    model_sd = {}
    for k in model.state_dict():
        if k in lycoris_sd:
            model_sd[k] = lycoris_sd.pop(k)
    model.load_state_dict(model_sd, strict=False)
    lycoris_net, _ = create_lycoris_from_weights(1.0, '', model, lycoris_sd)
    lycoris_net.half()
    lycoris_net.merge_to(1.0)
    model.cuda()
    model = torch.compile(model, mode='reduce-overhead')
    # lycoris_net.cuda()
    
    apply_attn_algo(model, algo='xformers')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    generate(model=model, tokenizer=tokenizer, prompt="Hello", max_new_tokens=16)

    print()
    while (request := input('Input your request: ')) != 'q':
        print('='*50, '\n')
        prev = ''
        prompt = f"""### Instruct:
Use student's preferation to predict the summary of final choosed course.

### Input:
{request}

### Response:
"""
        t0 = time_ns()
        for i in tqdm(
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=0.9,
                top_p=0.8,
                top_k=45,
                repetition_penalty=1.05,
                max_new_tokens=1024,
                stream_output=True,
            ), 
            disable=True
        ):
            if len(i) > len(prev):
                new_len = len(i) - len(prev)
                print(i[len(prev):], end='', flush=True)
                prev = i
            pass
        t1 = time_ns()
        result = i[len(prompt):]
        result_tokens = len(tokenizer.tokenize(result))
        print(i)
        print('='*50)
        print(f"Total generated tokens: {result_tokens}")
        print(f"Total cost time: {(t1-t0)/1e9:.2f}s")
        print(f"Average Speed: {(result_tokens/((t1-t0)/1e9)):.2f} tokens/sec")
        print()