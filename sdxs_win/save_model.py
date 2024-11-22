from diffusers import StableDiffusionPipeline
import torch

weight_type = torch.float32 # Или float16, но модель будет менее точной!


pipeline = StableDiffusionPipeline.from_pretrained('IDKiro/sdxs-512-0.9', torch_dtype=weight_type)

torch.save(pipeline, 'model.pth')
