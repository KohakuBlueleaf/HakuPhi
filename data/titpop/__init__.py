import os
from datasets import load_dataset
from random import shuffle, randint, choice, random


SELF_FOLDER = os.path.dirname(__file__)
SPLITS = {
    "danbooru": ["danbooru2023-prompt-gen-data.parquet"],
    "gbc": ["GBC10M-top-level-caption.parquet"],
}


def load(split="danbooru"):
    assert split in SPLITS
    return load_dataset(
        "parquet", data_files=[os.path.join(SELF_FOLDER, i) for i in SPLITS[split]]
    )["train"]


length_map = {
    "very_short": 10,
    "short": 20,
    "long": 40,
    "very_long": 60,
}
task_type = [
    "tag_to_long",
    "long_to_tag",
    "short_to_tag",
    "short_to_long",
    "tag_to_short_to_long",
    "short_to_tag_to_long",
    "short_to_long_to_tag",
    "gen_meta",
]
needed_special_token = [
    "<|empty|>",
    *(f"<|{length}|>" for length in length_map.keys()),
    *(f"<|{task}|>" for task in task_type),
]


def apply_special_tokens(tokenizer):
    tokenizer.add_tokens(needed_special_token, special_tokens=True)
    return tokenizer


def generate_prompt_dan(data, target_len="long", tag_seperator=", "):
    data["general"] = [
        tag for tag in data["general"] if (not (tag.isnumeric() and len(tag) == 4))
    ]
    total_target = len(data["general"])
    shuffle(data["general"])

    if target_len is None:
        target_len = [
            leng for leng, count_max in length_map.item() if count_max <= total_target
        ][-1]

    length = min(length_map[target_len], total_target)
    if length < length_map[target_len]:
        target_len = [i for i in length_map if length_map[i] <= length]
        target_len = "very_short" if not target_len else target_len[-1]
    input_target = randint(1, max(length * 3 // 5, 1) + 1)

    # 10% total drop
    total_drop = random() < 0.10
    if total_drop:
        input_target = 0

    prompt_input = data["special"] + data["general"][:input_target]
    prompt_output = data["general"][input_target:total_target]
    generals = data["special"] + data["general"]

    rating_tag = tag_seperator.join(data["rating"]) or "<|empty|>"
    artist_tag = tag_seperator.join(data["artist"]) or "<|empty|>"
    character_tag = tag_seperator.join(data["character"]) or "<|empty|>"
    copyright_tag = tag_seperator.join(data["copyright"]) or "<|empty|>"
    quality_tag = data.get("quality", None) or "<|empty|>"
    if isinstance(quality_tag, list) and len(quality_tag) > 0:
        quality_tag = quality_tag[0]

    drop_info = not total_drop and random() < 0.3
    if drop_info:
        rating_str = f"rating: {rating_tag if random() > 0.5 else '<|empty|>'}"
        artist_str = f"artist: {artist_tag if random() > 0.5 else '<|empty|>'}"
        character_str = (
            f"characters: {character_tag if random() > 0.5 else '<|empty|>'}"
        )
        copyright_str = (
            f"copyrights: {copyright_tag if random() > 0.5 else '<|empty|>'}"
        )
        quality_str = f"quality: {quality_tag if random() > 0.5 else '<|empty|>'}"
        if random() > 0.5:
            aspect_ratio = f"aspect ratio: {data['width']/data['height']:.1f}"
        else:
            aspect_ratio = f"aspect ratio: <|empty|>"
    else:
        # When total drop is triggered.
        # Provide all other information to learn the relationship
        rating_str = f"rating: {rating_tag}"
        artist_str = f"artist: {artist_tag}"
        character_str = f"characters: {character_tag}"
        copyright_str = f"copyrights: {copyright_tag}"
        quality_str = f"quality: {quality_tag}"
        aspect_ratio = f"aspect ratio: {data['width']/data['height']:.1f}"

    prior_info = [
        rating_str,
        artist_str,
        aspect_ratio,
        character_str,
        quality_str,
        copyright_str,
    ]
    shuffle(prior_info)
    prior = "\n".join(prior_info)
    florence_long = data["florence_long"]
    florence_short = data["florence_short"]
    phi3v_horny = data["phi3v_horny"]

    long = None
    short = florence_short
    if phi3v_horny is not None:
        long = phi3v_horny
    if florence_long is not None:
        if long is not None:
            short = florence_long if random() > 0.5 or short is None else short
        else:
            long = florence_long

    tasks = []
    if long is not None:
        tasks.extend(["tag_to_long", "long_to_tag"])

    # to not waste phi3v data
    if short is not None and phi3v_horny is None:
        tasks.extend(["short_to_tag"])

    if long is not None and short is not None:
        short = florence_short
        tasks.extend(
            ["tag_to_short_to_long", "short_to_long_to_tag", "short_to_tag_to_long"]
        )

    task = None
    if len(tasks) != 0 and random() < (len(tasks) / (len(tasks) + 1)):
        task = choice(tasks)
        if task.startswith("tag_to"):
            task_str = f"<|{target_len}|> <|{task}|>"
        else:
            task_str = f"<|{task}|>"
    else:
        task_str = f"<|{target_len}|>"

    full_data = {
        "tag": tag_seperator.join(generals),
        "short": short,
        "long": long,
    }

    output_prompt = ""
    addon_output_prompt = ""
    addon_user_prompt_before = ""
    addon_user_prompt_after = ""

    # 15% meta gen
    # 35% no meta
    # 50% normal
    meta_gen_mode = random()
    if meta_gen_mode < 0.15:
        task_str += " <|gen_meta|>"
        addon_output_prompt += "\n" + prior
    elif meta_gen_mode < 0.5:
        addon_user_prompt_before = ""
    else:
        addon_user_prompt_before = prior + "\n"

    if task is not None:
        data_order = task.split("_to_")
        if data_order[0] == "tag":
            addon_user_prompt_after += f"tag: {tag_seperator.join(prompt_input)}"
            output_prompt += tag_seperator + tag_seperator.join(prompt_output) + "\n"
        else:
            addon_user_prompt_after += f"{data_order[0]}: {full_data[data_order[0]]}"
            addon_user_prompt_after += "\n"

        for output_data in data_order[1:]:
            output_prompt += f"{output_data}: {full_data[output_data]}\n"
    else:
        addon_user_prompt_after += f"tag: {tag_seperator.join(prompt_input)}"
        output_prompt = tag_seperator + tag_seperator.join(prompt_output) + "\n"

    user_prompt = (
        addon_user_prompt_before + f"target: {task_str}\n" + addon_user_prompt_after
    )

    output_prompt = output_prompt.rstrip() + addon_output_prompt
    output_prompt = output_prompt.rstrip()

    # 30% train on input
    if random() < 0.7:
        user_prompt, output_prompt = "", user_prompt + output_prompt

    return user_prompt, output_prompt


def generate_prompt_gbc(data):
    short = data["short_caption"] if random() > 0.5 else data["original_caption"]
    long = data["detail_caption"]

    user_prompt = f"""target: <|short_to_long|>
short: {short}\nlong:""".strip()
    output_prompt = long

    if random() < 0.7:
        user_prompt, output_prompt = "", user_prompt + output_prompt

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
        if "original_caption" in data_point:
            user_part, output_part = generate_prompt_gbc(data_point)
        else:
            user_part, output_part = generate_prompt_dan(data_point)
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

    def load_tokenizer(
        tokenizer_ref="TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T",
    ):
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_ref)
        apply_special_tokens(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    tokenizer: LlamaTokenizer = load_tokenizer()
    dataset = load("all")
    data = choice(dataset)
    print(generate_prompt_dan(data, "very_long"))
    proc = processor(tokenizer, cutoff_len=384)
    print(proc(data))
