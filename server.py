import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# ------------- InternVL image preprocessing ------------- #

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


# ------------- Device + model load ------------- #

def choose_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


MODEL_NAME = "OpenGVLab/InternVL2-1B"
DEVICE = choose_device()

print(f"[server] Using device: {DEVICE}")
print("[server] Loading InternVL model... this is one-time at startup.")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False,
)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
).to(DEVICE).eval()

print("[server] Model loaded.")


# ------------- FastAPI app ------------- #

app = FastAPI(title="InternVL Scene Description API")


@app.post("/describe")
async def describe_scene(
    image: UploadFile = File(...),
    question: Optional[str] = Form("Please describe this scene in detail."),
):
    """
    Accepts an image (JPEG/PNG, etc) and an optional question.
    Returns JSON: { "description": "..." }
    """
    try:
        data = await image.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not read image: {e}"},
        )

    pixel_values = load_pixel_values_from_pil(pil_img, max_num=12).to(DEVICE)

    prompt = f"<image>\n{question}"
    gen_cfg = dict(max_new_tokens=256, do_sample=False)

    with torch.no_grad():
        answer = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=gen_cfg,
        )

    return {"description": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
