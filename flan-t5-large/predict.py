from http.client import HTTPException
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI()
model_ready = False

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/ready") == -1


class Item(BaseModel):
    input: str

# Filter out /ready
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

@app.get("/ready")
def get_readiness():
    if(model_ready):
        return {"model_ready": model_ready}
    else:
        raise HTTPException(status_code=500, detail="Model is not yet loaded. Please wait and try again later.")

@app.on_event("startup")
async def load_model():
    global model_ready
    print("Loading model...")
    global tokenizer, model
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", offload_folder="offload")
    model_ready = True
    print("Model has been loaded. Ready to serve requests.")

@app.post("/")
def predict(payload: Item):
    if not model_ready:
        return {"error": "Model is not yet loaded. Please wait and try again later."}

    input_ids = tokenizer(payload.input, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
