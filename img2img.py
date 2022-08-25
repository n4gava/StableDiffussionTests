from torch import autocast, float16
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DDIMScheduler
import PIL
from PIL import Image
import numpy as np
import uuid
import torch
from StableDiffusionImg2ImgPipeline import StableDiffusionImg2ImgPipeline
import requests
from io import BytesIO

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
torch.cuda.empty_cache()

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False, 
    set_alpha_to_one=False
)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=float16,
    scheduler=scheduler,
    use_auth_token=True
).to(device)

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.



url = "https://storage.googleapis.com/dream-machines-output/b4ebcc31-66b0-42f0-b3e4-da4322f610fb/0_0.png"

response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((512, 512))

init_image = preprocess_image(init_img)

prompt = "little humanoid tree, cute, big eyes, strong arms, realistic, unreal engine"
# generator = torch.Generator(device=device).manual_seed(1024)
with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5) #generator=generator)
    image = images["sample"][0]

image.save("teste2.png")