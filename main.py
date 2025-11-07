from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from predict.predict import predict_image

app = FastAPI(title="Image Recognition API")

model = MobileNetV2(weights="imagenet")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Image Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        result = await predict_image(model, file)
        
        return result
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})