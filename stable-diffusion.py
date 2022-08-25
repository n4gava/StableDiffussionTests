from torch import autocast, float16
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from PIL import Image
import uuid
import torch
import random

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


lms = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=float16,
    scheduler=lms,
    use_auth_token=True
).to(device)


def generateImage(prompt, num_imagens = 1, seed_input: int = 0, inference_steps_input: int = 0):
    imageFileName = f"{uuid.uuid4()}.png"
    seed = seed_input
    if (seed == 0):
        seed = random.randrange(1, 999999)

    inference_steps = inference_steps_input
    if (inference_steps == 0):
        inference_steps = 30

    print(f"Gerando imagem {imageFileName} com seed {seed} para o prompt: {prompt}")
    
    generator = torch.Generator("cuda").manual_seed(seed)

    with autocast(device):
        result = pipe([prompt] * num_imagens, 
            guidance_scale=7.5, 
            num_inference_steps=inference_steps,
            height=512, 
            width=512,
            generator=generator)

        samples = result["sample"]

    grid = image_grid(samples, rows=1, cols=num_imagens)
    grid.save(imageFileName)

while True:
    prompt = input("Descrição da imagem: ")
    seed = int(input("Seed: "))
    inference_steps = int(input("Steps: "))
    generateImage(prompt, 1, seed, inference_steps)
