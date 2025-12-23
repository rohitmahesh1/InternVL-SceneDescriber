import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from PIL import Image


# -------------------------
# InternVL2 preprocessing
# -------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


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

    grid_w = target_width // image_size

    for i in range(blocks):
        box = (
            (i % grid_w) * image_size,
            (i // grid_w) * image_size,
            ((i % grid_w) + 1) * image_size,
            ((i // grid_w) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_pixel_values_from_pil(pil_img: Image.Image, input_size=448, max_num=12) -> torch.Tensor:
    transform = build_transform(input_size)
    images = dynamic_preprocess(pil_img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


# -------------------------
# Device
# -------------------------

def choose_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -------------------------
# Handwriting crop (OpenCV)
# -------------------------

def naive_text_crop(pil_img: Image.Image):
    """
    Try to crop to the densest ink/text region.
    This is a heuristic 'best effort' crop — it greatly helps when the photo
    is mostly a sheet of paper or a note.
    If OpenCV isn't available, returns original image.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return pil_img

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase chance of isolating handwriting strokes
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img

    # Find largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Avoid absurd crops (e.g., tiny UI icons)
    H, W = gray.shape[:2]
    if w * h < 0.02 * (W * H):
        return pil_img

    pad = 15
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)

    return pil_img.crop((x, y, x + w, y + h))


# -------------------------
# InternVL yes/no gate
# -------------------------

def internvl_yesno(model, tokenizer, pixel_values, question_text):
    prompt = f"<image>\n{question_text}\nAnswer only yes or no."
    gen_cfg = dict(max_new_tokens=5, do_sample=False)
    with torch.no_grad():
        resp = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=gen_cfg
        )
    resp = resp.strip().lower()
    return resp.startswith("yes")


# -------------------------
# TrOCR
# -------------------------

def trocr_transcribe(trocr_model, trocr_processor, pil_img, device):
    inputs = trocr_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = trocr_model.generate(**inputs, max_new_tokens=256)
    text = trocr_processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return text.strip()


def looks_bad_ocr(text: str):
    t = text.strip()
    if len(t) < 4:
        return True
    letters = sum(ch.isalpha() for ch in t)
    return letters / max(len(t), 1) < 0.3


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Please describe this image in detail.")
    parser.add_argument("--internvl-model", default="OpenGVLab/InternVL2-1B")
    parser.add_argument("--trocr-model", default="microsoft/trocr-large-handwritten")
    parser.add_argument("--max-num", type=int, default=12)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    device = choose_device()
    print(f"Using device: {device}")

    pil_img = Image.open(str(image_path)).convert("RGB")

    # Load InternVL
    print("Loading InternVL tokenizer/model...")
    internvl_tokenizer = AutoTokenizer.from_pretrained(
        args.internvl_model, trust_remote_code=True, use_fast=False
    )
    internvl_model = AutoModel.from_pretrained(
        args.internvl_model, trust_remote_code=True
    ).to(device).eval()

    # Preprocess for InternVL
    pixel_values = load_pixel_values_from_pil(pil_img, max_num=args.max_num).to(device)

    # Handwriting gate
    print("Checking for handwritten text...")
    has_handwriting = internvl_yesno(
        internvl_model,
        internvl_tokenizer,
        pixel_values,
        "Does this image contain handwritten text that should be transcribed?"
    )

    if has_handwriting:
        print("Handwriting detected → enabling TrOCR with smart crop...")

        # Load TrOCR
        trocr_processor = TrOCRProcessor.from_pretrained(args.trocr_model)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model).to(device).eval()

        # Crop for better OCR
        cropped = naive_text_crop(pil_img)

        text = trocr_transcribe(trocr_model, trocr_processor, cropped, device)

        # Fallback if TrOCR output is weak
        if looks_bad_ocr(text):
            print("TrOCR output looks weak → falling back to InternVL OCR prompt...")
            ocr_prompt = "<image>\nPlease transcribe the handwritten text exactly as written."
            gen_cfg = dict(max_new_tokens=256, do_sample=False)

            with torch.no_grad():
                text = internvl_model.chat(
                    tokenizer=internvl_tokenizer,
                    pixel_values=pixel_values,
                    question=ocr_prompt,
                    generation_config=gen_cfg
                ).strip()

        print("\n=== Result ===")
        print("Mode: handwriting_ocr")
        print("Extracted text:", text if text else "(No text recognized)")
        return

    # Default vision chat
    print("No handwriting detected → using InternVL for vision chat...")
    prompt = f"<image>\n{args.question}"
    gen_cfg = dict(max_new_tokens=256, do_sample=False)

    with torch.no_grad():
        resp = internvl_model.chat(
            tokenizer=internvl_tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=gen_cfg
        )

    print("\n=== Result ===")
    print("Mode: vision_chat")
    print("User:", args.question)
    print("Assistant:", resp)


if __name__ == "__main__":
    main()
