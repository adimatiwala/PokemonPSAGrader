from train_model import CardGradingModel, get_transforms
import torch
from extract_card import extract_card
import argparse
from pathlib import Path

# Runs the user input through the model
@torch.no_grad()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) instantiate model
    model = CardGradingModel(num_classes=10).to(device)

    # 2) load checkpoint (Here is where you put your model)
    ckpt       = torch.load('best_model.pth', map_location=device)
    sd         = ckpt.get('state_dict', ckpt)
    model_sd   = model.state_dict()

    filtered = {
        k: v for k, v in sd.items()
        if k in model_sd and model_sd[k].shape == v.shape
    }

    # load
    model.load_state_dict(filtered, strict=False)
    model.eval()

    # 4) use inference transforms
    tfm = get_transforms(is_training=False)
    
    # Prep image
    img = extract_card(str(args.image), download=False)
    tensor = tfm(image=img)["image"].unsqueeze(0).to(device)

    # Inference
    logits = model(tensor)
    probs  = logits.softmax(dim=1).squeeze().cpu().numpy()

    grade = int(probs.argmax() + 1)  # map 0–9 → PSA 1–10
    print(f"Predicted PSA grade: {grade}")
    print("Full probability distribution (PSA 1-10):", [round(p,3) for p in probs])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",   type=Path, required=True, help="path to card photo (JPG/PNG)")
    main(p.parse_args())