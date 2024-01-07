"""
Add new tokens from Traditional Chinese and Japanese
"""
import os
import csv

from transformers import AutoTokenizer
from hakuphi.model import PhiForCausalLM


def load_model(path="microsoft/phi-1_5"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = PhiForCausalLM.from_pretrained(path, trust_remote_code=True)
    return tokenizer, model


def load_extra_tokens(path="./extra_tokens"):
    extra_tokens = set()
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == ".csv":
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                first_row = next(reader)
                token_pos = int(first_row[0])
                extra_tokens |= set(
                    [row[token_pos].split("（")[0] for row in reader if row[token_pos]]
                )
        elif os.path.splitext(file)[1] == ".txt":
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                extra_tokens |= set(f.read().split())
        else:
            print(f"ignore {file}: unsupported file type")
            continue
    extra_tokens.discard("")
    return sorted(extra_tokens)


def main():
    test_extra_tokens = load_extra_tokens()
    tokenizer, model = load_model()

    tokenizer.add_tokens(test_extra_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(model.modules[0].wte)


if __name__ == "__main__":
    main()
