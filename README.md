# InternVL Scene Describer Server

This repo is based on [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) and extends it with a **scene description / visual QA server** that you can call from an iOS app over gRPC.

The core idea:  
> Take an image → run it through `OpenGVLab/InternVL3-1B` → return a natural-language description or an answer to a visual question.

I use this in a companion iOS app (“SceneDescriber”) that captures photos with the iPhone camera and calls this server via gRPC.

---

## Features

- ✅ **InternVL3-1B model wrapper**

  - Loads `OpenGVLab/InternVL3-1B` via Hugging Face `transformers`  
  - Uses Apple Metal (`mps`) on macOS if available, otherwise CUDA / CPU  
  - Handles dynamic image preprocessing and multi-patch inputs

- ✅ **gRPC API (Python)** — defined in `scenedescriber.proto`

  Implemented in `server_grpc.py`:

  - `SayHello(name)` → simple hello test (good for connectivity)
  - `DescribeScene(image)` → long-form description: “what’s going on in this picture?”
  - `AskAboutScene(image, question)` → visual question answering
  - `OcrImage(image)` → placeholder endpoint for future OCR integration (DeepSeekOCR / TrOCR)

- ✅ **REST / HTTP demo (optional)**

  - `server.py` exposes a FastAPI endpoint for simple `curl` / browser testing

- ✅ **Example clients**

  - `demo_image_chat.py` – simple CLI demo of image + question ↦ text answer  
  - `smart_image_chat.py` – more advanced example (e.g., question templates)  
  - `test_falls.py` – test script for a specific “waterfall” image and tuning generation params

---

## Repo structure (relevant files)

Only the key extensions are listed here; the rest is the original InternVL codebase.

```text
InternVL-SceneDescriber/
├─ scenedescriber.proto          # gRPC service + message definitions
├─ scenedescriber_pb2.py         # Generated Python messages (protoc)
├─ scenedescriber_pb2_grpc.py    # Generated Python service stubs
├─ server_grpc.py                # gRPC server wrapping InternVL3-1B
├─ server.py                     # Optional FastAPI HTTP server
├─ demo_image_chat.py            # Simple image+question demo
├─ smart_image_chat.py           # Slightly higher-level demo
├─ test_falls.py                 # Example test for a specific image
├─ Steps.md                      # Personal notes / setup steps
└─ requirements/
   └─ internvl_chat.txt          # Python dependencies (extended for this project)
