from diffusers import StableDiffusionPipeline
import torch, os
import uuid
import numpy as np
from config import numImages, numDenoisingSteps

def makeImage(phrase : str, dire : str):
    generator = torch.Generator(device="cuda").manual_seed(466)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(current_dir, "../images")

    # experimental speedup
    torch.backends.cuda.matmul.allow_tf32 = True

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.to("cuda")

    if not os.path.exists(os.path.join(out_dir, dire)):
        os.makedirs(os.path.join(out_dir, dire))

    for _ in range(numImages):
        image = pipe(phrase, 
                     negative_prompt="blurry, cropped, bad anatomy, worst quality, error, text, watermark", 
                     generator=generator,
                     num_inference_steps=numDenoisingSteps,
                     guidance_scale=np.random.randint(2, 9)).images[0]
        image.save(os.path.join(out_dir, dire, str(uuid.uuid4()) + '.png'))

if __name__ == "__main__":
    c = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for s in c:
        makeImage(s, s)