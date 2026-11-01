import torch

def generate_response(model, question, device, tokenizer):
    model.eval()

    prompt = f"Question:\n{question}\n\nAnswer:\n"

    input_ids = torch.tensor(
        [tokenizer.encode_ordinary(prompt)],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eot_token,
            eos_token_id=tokenizer.eot_token,
        )

    # Decode only newly generated tokens
    generated_ids = outputs[0, input_ids.shape[1]:].tolist()

    # Remove EOS and anything after it
    if tokenizer.eot_token in generated_ids:
        generated_ids = generated_ids[
            :generated_ids.index(tokenizer.eot_token)
        ]

    return tokenizer.decode(generated_ids).strip()