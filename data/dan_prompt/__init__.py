import os
from datasets import load_dataset
from random import shuffle, randint, choice, random


SELF_FOLDER = os.path.dirname(__file__)
SPLITS = {
    "all": ["captions.jsonl"],
}


def load(split="all"):
    assert split in SPLITS
    return load_dataset(
        "json", data_files=[os.path.join(SELF_FOLDER, i) for i in SPLITS[split]]
    )["train"]


length_map = {
    "very_short": 10,
    "short": 20,
    "long": 40,
    "very_long": 80,
}
needed_special_token = [
    "<|input_end|>",
    "<|empty|>",
    *(f"<|{length}|>" for length in length_map.keys()),
]


def apply_special_tokens(tokenizer):
    tokenizer.add_tokens(needed_special_token, special_tokens=True)
    return tokenizer


def generate_prompt(data, target_len="long", tag_seperator=", "):
    if target_len is None:
        target_len = choice(length_map)
    shuffle(data["general"])

    total_target = len(data["general"])
    length = min(length_map[target_len], total_target)
    input_target = randint(
        1 * (length // 5),
        length - 1 * (length // 5),
    )

    prompt_input = data["general"][:input_target]
    prompt_output = data["general"][input_target:total_target]
    generals = data["special"] + prompt_input

    rating_str = f"rating: {tag_seperator.join(data['rating']) or '<|empty|>' if random() > 0.7 else '<|empty|>'}"
    artist_str = f"artist: {tag_seperator.join(data['artist']) or '<|empty|>' if random() > 0.7 else '<|empty|>'}"
    character_str = f"characters: {tag_seperator.join(data['character']) or '<|empty|>' if random() > 0.7 else '<|empty|>'}"
    copyright_str = f"copyrights: {tag_seperator.join(data['copyright']) or '<|empty|>' if random() > 0.7 else '<|empty|>'}"

    prior_info = [rating_str, artist_str, character_str, copyright_str]
    shuffle(prior_info)
    prior = "\n".join(prior_info)

    user_prompt = f"""{prior}
target: <|{target_len}|>
general: {tag_seperator.join(generals)}<|input_end|>""".strip()

    output_prompt = tag_seperator.join(prompt_output)

    return user_prompt, output_prompt


def tokenize(tokenizer, prompt, cutoff_len=2048, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        # return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def processor(tokenizer, cutoff_len=2048, train_on_inputs=False, padding=True):
    import torch

    def generate_and_tokenize_prompt(data_point):
        user_part, output_part = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(
            tokenizer, user_part + output_part, cutoff_len, add_eos_token=True
        )
        tokenized_user_prompt = tokenize(
            tokenizer, user_part, cutoff_len, add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        full_prompt_len = len(tokenized_full_prompt["input_ids"])

        if not train_on_inputs:
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        pad_len = cutoff_len - full_prompt_len
        if padding:
            tokenized_full_prompt["input_ids"] = (
                tokenized_full_prompt["input_ids"] + [0] * pad_len
            )
            tokenized_full_prompt["labels"] = (
                tokenized_full_prompt["labels"] + [-100] * pad_len
            )
            tokenized_full_prompt["attention_mask"] = (
                tokenized_full_prompt["attention_mask"] + [0] * pad_len
            )

        for k in tokenized_full_prompt.keys():
            tokenized_full_prompt[k] = torch.LongTensor(tokenized_full_prompt[k])
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


if __name__ == "__main__":
    from transformers import LlamaTokenizer
    def load_tokenizer(tokenizer_ref="TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"):
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_ref)
        apply_special_tokens(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    tokenizer: LlamaTokenizer = load_tokenizer()
    dataset = load("all")
    data = choice(dataset)
    print(generate_prompt(data, "very_long"))
    proc = processor(tokenizer, cutoff_len=384)
    print(proc(data))
