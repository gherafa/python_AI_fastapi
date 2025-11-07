# python_AI_fastapi
# Image Recognition API (FastAPI + TensorFlow)

A lightweight **Image Recognition REST API** built with **FastAPI** and **TensorFlow (MobileNetV2)**.  
It accepts an uploaded image and returns top-3 classification predictions from the pretrained ImageNet model.

---

## Features
- FastAPI backend (async + auto docs)
- TensorFlow MobileNetV2 pre-trained model
- Image upload endpoint (`/predict`)
- CORS enabled for frontend (React/Next.js ready)
- Unit tests with `pytest`
- Modular code structure (`predict/` module)