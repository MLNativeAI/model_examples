FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
WORKDIR app/
COPY . ./
ENV TRANSFORMERS_CACHE /workspace/app/cache
RUN pip install --no-cache-dir -r ./requirements.txt
RUN python download.py
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0"]