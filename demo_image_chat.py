import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from PIL import Image


# ---- Image preprocessing (adapted from InternVL2 docs, simplified for single image) ----

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file: str, input_size=448, max_num=12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)  # shape: (num_patches, 3, H, W)
    return pixel_values


# ---- Main demo ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument(
        "--question",
        default="Please describe this image in detail.",
        help="Question to ask about the image",
    )
    parser.add_argument(
        "--model-path",
        default="OpenGVLab/InternVL2-1B",
        help="Hugging Face model name or local path",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    # Choose device: prefer Apple GPU (mps), else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    model_path = args.model_path

    print("Loading tokenizer and model (this may take a bit the first time)...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).to(device).eval()

    print(f"Loading and preprocessing image: {image_path}")
    pixel_values = load_image(str(image_path), max_num=12).to(device)

    # Prompt format expected by InternVL2: include <image> token
    question = f"<image>\n{args.question}"
    generation_config = dict(max_new_tokens=256, do_sample=False)

    print("Running model.chat()...")
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
        )

    print("\n=== Chat ===")
    print("User:", args.question)
    print("Assistant:", response)


if __name__ == "__main__":
    main()
