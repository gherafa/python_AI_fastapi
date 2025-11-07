from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

async def predict_image(model, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    
    top_predictions = [
        {"label": label, "confidence": float(conf)}
        for (_, label, conf) in decoded
    ]
    
    main_pred = top_predictions[0]
    
    return JSONResponse(content={
        "filename": file.filename,
        "prediction": main_pred["label"],
        "confidence": main_pred["confidence"],
        "top_predictions": top_predictions
    })