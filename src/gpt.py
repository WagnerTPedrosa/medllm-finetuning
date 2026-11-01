import torch
from transformers import AutoModelForCausalLM


def load_model(model: str):
    # [Step 1] Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # [Step 2] Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(model)

    model.config.use_cache = False
    model.to(device)

    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    return model, device
