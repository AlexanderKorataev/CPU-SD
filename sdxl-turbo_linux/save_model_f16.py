from diffusers import AutoPipelineForText2Image
import torch

weight_type = torch.float16


pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=weight_type, variant="fp16")

torch.save(pipeline, 'model_f16.pth')
