import torch

def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            # Count target tokens used in next-token loss
            num_tokens = (
                batch["labels"][:, 1:] != -100
            ).sum().item()

            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens