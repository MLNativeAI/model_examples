from http.client import HTTPException

from TTS.api import TTS
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/ready") == -1

class Item(BaseModel):
    input: str

# Filter out /ready
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI()
model_ready = False

@app.get("/ready")
def get_readiness():
    if(model_ready):
        return {"model_ready": model_ready}  # Step 2: Expose the readiness status
    else:
        raise HTTPException(status_code=500, detail="Model is not yet loaded. Please wait and try again later.")

@app.on_event("startup")
async def load_model():
    global model_ready
    print("Loading model...")
    global tts
    tts = TTS('tts_models/en/ek1/tacotron2')
    model_ready = True
    print("Model has been loaded. Ready to serve requests.")

@app.post("/")
def predict(payload: Item):
    if not model_ready:
        return {"error": "Model is not yet loaded. Please wait and try again later."}

    tts.tts_to_file(text=payload.input, file_path='/data/sound.wav')

    # Return sound file as a streaming response
    return FileResponse('/data/sound.wav')
