import os
from datasets import load_dataset
from random import shuffle, randint, choice


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

    total_target = length_map[target_len] - len(data["special"])
    input_target = randint(1, length_map[target_len] - 1)

    prompt_input = data["general"][:input_target]
    prompt_output = data["general"][input_target:total_target]
    generals = data["special"] + prompt_input
    prompt = f"""
rating: {tag_seperator.join(data["rating"]) or "<|empty|>"}
artist: {tag_seperator.join(data["artist"]) or "<|empty|>"}
characters: {tag_seperator.join(data["character"]) or "<|empty|>"}
copyrights: {tag_seperator.join(data["copyright"]) or "<|empty|>"}
target: <|{target_len}|>
general: {tag_seperator.join(generals)}<|input_end|>
{tag_seperator.join(prompt_output)}""".strip()

    return prompt


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


def processor(tokenizer, cutoff_len=512, padding=True):
    import torch

    def generate_and_tokenize_prompt(data_point):
        prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(
            tokenizer, prompt, cutoff_len, add_eos_token=True
        )

        pad_len = cutoff_len - len(tokenized_full_prompt["input_ids"])
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
    dataset = load("all")
    data = next(iter(dataset["train"]))
    print(generate_prompt(data, "very_long"))
