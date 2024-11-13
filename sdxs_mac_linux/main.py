import torch
import diffusers
import transformers
import os
import re
from PIL import Image
import gc
import numpy


model_path = "./model.pth"
pipe = torch.load(model_path, weights_only=False)
device = "mps"
pipe.to(device)

def get_image_path(prompt):
    base_name = prompt.replace(" ", "_")
    pattern = re.compile(rf"{re.escape(base_name)}_(\d{{5}})\.png")
    
    os.makedirs("./images", exist_ok=True)
    existing_files = [f for f in os.listdir("./images") if pattern.match(f)]
    
    max_number = 0
    for file_name in existing_files:
        match = pattern.match(file_name)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    
    new_number = max_number + 1
    file_path = f"./images/{base_name}_{new_number:05d}.png"
    
    return file_path

    

while True:
    prompt = input("Введите промпт для генерации изображения ('Ctrl + C' для выхода):")
    if not prompt:
        print("Промпт не может быть пустым. Попробуйте снова.")
        continue
    
    with torch.no_grad(): 
        image = pipe(prompt, height=512, width=512, num_inference_steps=1, guidance_scale=0).images[0]
    
    image_path = get_image_path(prompt)

    image.save(image_path)
    print(f"Изображение сохранено в {image_path}")

    del image
    gc.collect()