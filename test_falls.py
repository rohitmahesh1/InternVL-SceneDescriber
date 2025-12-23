import io
import argparse
from pathlib import Path

from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModel

from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

# --------- Config: match your server's model here ---------

MODEL_NAME = "OpenGVLab/InternVL3-1B"  # change if you're using a different one

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    """Simple resize + normalize like in server_grpc.py."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def prepare_pixel_values(path: Path, input_size: int = 448, device: str = "cpu"):
    img = Image.open(path).convert("RGB")
    transform = build_transform(input_size)
    tensor = transform(img).unsqueeze(0)  # shape: (1, 3, H, W)
    return tensor.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file (e.g. ~/Downloads/falls.png)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Hugging Face model name or local path (default: OpenGVLab/InternVL3-1B)",
    )
    parser.add_argument(
        "--question",
        default="Please describe this scene in detail.",
        help="Question to ask about the image (DescribeScene uses this by default).",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    # Device selection: prefer MPS, then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"[test_describe_scene] Using device: {device}")
    print(f"[test_describe_scene] Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
    ).to(device).eval()

    print(f"[test_describe_scene] Loading and preprocessing image: {image_path}")
    pixel_values = prepare_pixel_values(image_path, input_size=448, device=device)

    # This matches your DescribeScene behavior (no explicit question in the proto)
    full_question = f"<image>\n{args.question}"

    generation_config = dict(
        max_new_tokens=512,   # plenty of room for the waterfall paragraph
        do_sample=False,
    )

    print("[test_describe_scene] Running model.chat()...")
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=full_question,
            generation_config=generation_config,
        )

    print("\n========== MODEL RESPONSE ==========")
    print(response)
    print("============= END ==================")


if __name__ == "__main__":
    main()
