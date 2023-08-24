import base64
from http.client import HTTPException

from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import logging

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/ready") == -1

# Filter out /ready
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(
    max_upload_size=4 * 1024 * 1024  # 4MB
)
model_ready = False

@app.on_event("startup")
async def load_model():
    global model_ready
    print("Loading model...")
    global processor, model
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model_ready = True
    print("Model has been loaded. Ready to serve requests.")

class Item(BaseModel):
    input: str

@app.get("/ready")
def get_readiness():
    if(model_ready):
        return {"model_ready": model_ready}  # Step 2: Expose the readiness status
    else:
        raise HTTPException(status_code=500, detail="Model is not yet loaded. Please wait and try again later.")

@app.post("/")
def predict(payload: Item):
    if not model_ready:
        return {"error": "Model is not yet loaded. Please wait and try again later."}

    # Decode the base64 image bytes and convert to PIL Image
    image_bytes = base64.b64decode(payload.input)
    image = Image.open(BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]



