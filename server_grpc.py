# server_grpc.py

import io
from concurrent import futures
from typing import Optional

import grpc
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModel

import scenedescriber_pb2
import scenedescriber_pb2_grpc

from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from grpc_reflection.v1alpha import reflection


# ---------- InternVL model loading ----------

MODEL_NAME = "OpenGVLab/InternVL3-1B"

print("[gRPC server] Loading InternVL model...")

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False,
)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
).to(device).eval()

print(f"[gRPC server] Model loaded on device: {device}")


# ---------- Image → pixel_values helper ----------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def prepare_pixel_values_from_bytes(image_bytes: bytes, input_size: int = 448):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = build_transform(input_size)
    tensor = transform(img).unsqueeze(0)  # shape (1, 3, H, W)
    return tensor.to(device)


# ---------- InternVL call helpers ----------

def run_internvl_scene(image_bytes: bytes, question: Optional[str] = None) -> str:
    """Call InternVL3 to describe or answer about a scene."""
    if not question:
        question = "Please describe this scene in detail."

    pixel_values = prepare_pixel_values_from_bytes(image_bytes)
    full_question = f"<image>\n{question}"

    generation_config = dict(
        max_new_tokens=1024,
        do_sample=False,
    )

    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=full_question,
            generation_config=generation_config,
        )

    return response


def run_ocr_stub(image_bytes: bytes) -> str:
    """
    Placeholder OCR implementation.
    Later you can:
      - route to DeepSeek OCR via vLLM, or
      - use TrOCR from Hugging Face.
    For now, we just return a stub string.
    """
    return "OCR not implemented yet. This is where DeepSeekOCR or TrOCR would run."


# ---------- gRPC service implementation ----------

class SceneServiceServicer(scenedescriber_pb2_grpc.SceneServiceServicer):
    """Implements the SceneService defined in scenedescriber.proto"""

    def SayHello(self, request, context):
        name = request.name.strip() or "there"
        msg = f"Hello, {name}! Welcome to SceneDescriber over gRPC."
        return scenedescriber_pb2.GreetReply(message=msg)

    def DescribeScene(self, request, context):
        try:
            print(f"[gRPC] DescribeScene called (filename={request.filename})")
            text = run_internvl_scene(request.image, question=None)
            return scenedescriber_pb2.SceneReply(text=text)
        except Exception as e:
            err = f"Error describing scene: {e}"
            print("[gRPC] ", err)
            context.set_details(err)
            context.set_code(grpc.StatusCode.INTERNAL)
            return scenedescriber_pb2.SceneReply()

    def AskAboutScene(self, request, context):
        try:
            print(f"[gRPC] AskAboutScene called (filename={request.filename})")
            q = request.question or "Please describe this scene."
            text = run_internvl_scene(request.image, question=q)
            return scenedescriber_pb2.SceneReply(text=text)
        except Exception as e:
            err = f"Error answering question: {e}"
            print("[gRPC] ", err)
            context.set_details(err)
            context.set_code(grpc.StatusCode.INTERNAL)
            return scenedescriber_pb2.SceneReply()

    def OcrImage(self, request, context):
        try:
            print(f"[gRPC] OcrImage called (filename={request.filename})")
            text = run_ocr_stub(request.image)
            return scenedescriber_pb2.OcrReply(text=text)
        except Exception as e:
            err = f"Error running OCR: {e}"
            print("[gRPC] ", err)
            context.set_details(err)
            context.set_code(grpc.StatusCode.INTERNAL)
            return scenedescriber_pb2.OcrReply()


# ---------- server bootstrap ----------

def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    # Register your service
    scenedescriber_pb2_grpc.add_SceneServiceServicer_to_server(
        SceneServiceServicer(), server
    )

    # 🔹 Enable server reflection
    SERVICE_NAMES = (
        scenedescriber_pb2.DESCRIPTOR.services_by_name["SceneService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[gRPC server] Listening on 0.0.0.0:{port} with reflection enabled")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
