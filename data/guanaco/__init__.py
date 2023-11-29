import os
from datasets import load_dataset


SELF_FOLDER = os.path.dirname(__file__)
SPLITS = {
    'all': ['guanaco_chat_all.json', 'guanaco_non_chat_json'],
    'non-chat': ['guanaco_non_chat.json'],
    'mini': ['guanaco_non_chat_mini_52K.json'],
    'test': ['test.json'],
}


def load(split='all'):
    assert split in SPLITS
    return load_dataset(
        'json', 
        data_files=[
            os.path.join(SELF_FOLDER, i) 
            for i in SPLITS[split]
        ]
    )


def generate_prompt(data_point):
    '''Guanaco-alpaca chat format'''
    if data_point["input"]:
        user_part = f"""### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

"""
    else:
        user_part = f"""### Instruction:
{data_point["instruction"]}

"""
    
    output_part = f"""### Response:
{data_point["output"]}"""

    return user_part, output_part


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
        tokenized_full_prompt = tokenize(tokenizer, user_part+output_part, cutoff_len, add_eos_token=True)
        tokenized_user_prompt = tokenize(tokenizer, user_part, cutoff_len, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        full_prompt_len = len(tokenized_full_prompt["input_ids"])
        
        if not train_on_inputs:
            tokenized_full_prompt['labels'] = (
                [-100] * user_prompt_len 
                + tokenized_full_prompt['labels'][user_prompt_len:]
            )
        
        pad_len = cutoff_len - full_prompt_len
        if padding:
            tokenized_full_prompt['input_ids'] = (
                tokenized_full_prompt['input_ids']
                + [0] * pad_len 
            )
            tokenized_full_prompt['labels'] = (
                tokenized_full_prompt['labels']
                + [-100] * pad_len
            )
            tokenized_full_prompt['attention_mask'] = (
                tokenized_full_prompt['attention_mask']
                + [0] * pad_len 
            )
        
        for k in tokenized_full_prompt.keys():
            tokenized_full_prompt[k] = torch.LongTensor(tokenized_full_prompt[k])
        return tokenized_full_prompt
    
    return generate_and_tokenize_prompt