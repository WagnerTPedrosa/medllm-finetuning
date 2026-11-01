import tiktoken
import torch
from torch.nn.utils.rnn import pad_sequence

tokenizer = tiktoken.get_encoding("gpt2")


def tokenize_dataset(dataset):
    tokenized_data = []

    for example in dataset:
        prompt = f"Question:\n{example['prompt']}\n\nAnswer:\n"
        completion = example["completion"]

        prompt_ids = tokenizer.encode_ordinary(prompt)
        completion_ids = tokenizer.encode_ordinary(completion)

        completion_ids.append(tokenizer.eot_token)

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        tokenized_data.append({"input_ids": input_ids, "labels": labels})

    return tokenized_data


def collate_fn(batch):
    # [Step 1] Convert lists to PyTorch tensors
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]

    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

    # [Step 2] Create attention masks
    attention_masks = [torch.ones(len(ids), dtype=torch.long) for ids in input_ids]

    # [Step 3] Apply dynamic padding
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.eot_token
    )

    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}
