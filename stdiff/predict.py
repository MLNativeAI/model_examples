from http.client import HTTPException

from fastapi import FastAPI, Response
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
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
    if (model_ready):
        return {"model_ready": model_ready}  # Step 2: Expose the readiness status
    else:
        raise HTTPException(status_code=500, detail="Model is not yet loaded. Please wait and try again later.")


@app.on_event("startup")
async def load_model():
    global model_ready, model
    print("Loading model...")
    model = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir="/workspace/app/cache", torch_dtype=torch.float16)
    model = model.to("cuda")
    model.enable_attention_slicing()
    model.enable_model_cpu_offload()
    model.enable_xformers_memory_efficient_attention()
    model_ready = True
    print("Model has been loaded. Ready to serve requests.")


@app.post("/",
          responses={
              200: {
                  "content": {"image/png": {}}
              }
          },

          # Prevent FastAPI from adding "application/json" as an additional
          # response media type in the autogenerated OpenAPI specification.
          # https://github.com/tiangolo/fastapi/issues/3258
          response_class=Response)
def predict(payload: Item):
    input_text = payload.input
    image = model(input_text, guidance_scale=7.5).images[0]
    img_io = BytesIO()
    image.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return Response(img_io.getvalue(), media_type="image/png")
