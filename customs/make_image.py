from diffusers import StableDiffusionPipeline
import torch, os
import uuid


def makeImage(phrase : str, dire : str):
    device = "cuda"
    generator = torch.Generator(device="cuda").manual_seed(466)

    out_dir = "../images"
    model_path = "../models/pytorch_lora_weights.bin"

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", #change to 2-1 eventually
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.unet.load_attn_procs(model_path)
    pipe.to(device)
    os.system("mkdir " + os.path.join(out_dir, dire))
    for _ in range(4):
        image = pipe(phrase, num_inference_steps=30, generator=generator).images[0]
        image.save(os.path.join(out_dir, dire, str(uuid.uuid4()) + '.png'))

if __name__ == "__main__":
    makeImage("hello")