from diffusers import DiffusionPipeline

DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", cache_dir="/workspace/app/cache"
)

print("Download complete")
