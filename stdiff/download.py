from diffusers import DiffusionPipeline

DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir="/workspace/app/cache")

print("Download complete")