import pytest
from io import BytesIO
from PIL import Image
from fastapi import UploadFile

import predict.predict as predict_module
from predict.predict import predict_image

class DummyModel:
    def predict(self, img_array):
        import numpy as np

        return np.zeros((1, 1000))

@pytest.fixture(autouse=True)
def patch_decode(monkeypatch):
    def fake_decode(preds, top=3):
        return [[("n1", "cat", 0.8), ("n2", "dog", 0.15), ("n3", "car", 0.05)]]

    monkeypatch.setattr(predict_module, "decode_predictions", fake_decode)

def create_test_image() -> UploadFile:
    img = Image.new("RGB", (224, 224), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    return UploadFile(filename="test.jpg", file=buf)

@pytest.mark.asyncio
async def test_predict_image():
    model = DummyModel()
    upload_file = create_test_image()

    response = await predict_image(model, upload_file)
    body = response.body.decode()
    
    assert "cat" in body
    assert '"prediction"' in body
    assert '"confidence"' in body