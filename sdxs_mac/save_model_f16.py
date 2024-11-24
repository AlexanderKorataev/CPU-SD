from diffusers import StableDiffusionPipeline
import torch

weight_type = torch.float16


pipeline = StableDiffusionPipeline.from_pretrained('IDKiro/sdxs-512-0.9', torch_dtype=weight_type)

torch.save(pipeline, 'model_f16.pth')
