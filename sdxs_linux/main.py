import argparse
import torch
from diffusers import StableDiffusionPipeline
import os
import sys
import gc
import time
import psutil
from PIL import Image


def save_image_to_file(image, prompt):
    base_name = prompt.replace(" ", "_").strip()
    output_dir = "./images"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{base_name}.png")
    counter = 1

    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{base_name}_{counter}.png")
        counter += 1

    image.save(file_path, format="PNG")
    return file_path

def main():
    parser = argparse.ArgumentParser(description="Генерация изображений с использованием модели Stable Diffusion.")
    parser.add_argument("model_path", type=str, help="Путь к файлу модели.")

    args = parser.parse_args()

    pipe = torch.load(args.model_path, weights_only=False)
    pipe.set_progress_bar_config(disable=True)
    
    device = "cpu"
    pipe.to(device)


    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:  
            sys.exit(0)
        
        try:
            start_time = time.time()
            memory_before = psutil.virtual_memory().used

            with torch.no_grad():
                image = pipe(prompt, height=512, width=512, num_inference_steps=1, guidance_scale=0).images[0]
            
            memory_after = psutil.virtual_memory().used
            generation_time = time.time() - start_time

            print(f"Генерация завершена: {generation_time:.2f} секунд, "
                  f"использовано памяти: {((memory_after - memory_before) / 1e6):.2f} MB.",
                  file=sys.stderr)

            file_path = save_image_to_file(image, prompt)

            print(file_path)

            del image
            gc.collect()
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
