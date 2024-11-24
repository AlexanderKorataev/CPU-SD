# import torch
# import diffusers
# import transformers
# import os
# import re
# from PIL import Image
# import gc
# import numpy
# import requests



# model_path = "./model.pth"
# pipe = torch.load(model_path, weights_only=False)
# device = "mps"
# pipe.to(device)

# print(pipe)

# def get_image_path(prompt):
#     base_name = prompt.replace(" ", "_")
#     pattern = re.compile(rf"{re.escape(base_name)}_(\d{{5}})\.png")
    
#     os.makedirs("./images", exist_ok=True)
#     existing_files = [f for f in os.listdir("./images") if pattern.match(f)]
    
#     max_number = 0
#     for file_name in existing_files:
#         match = pattern.match(file_name)
#         if match:
#             number = int(match.group(1))
#             if number > max_number:
#                 max_number = number
    
#     new_number = max_number + 1
#     file_path = f"./images/{base_name}_{new_number:05d}.png"
    
#     return file_path

    

# while True:
#     prompt = input("Введите промпт для генерации изображения ('Ctrl + C' для выхода):")
#     if not prompt:
#         print("Промпт не может быть пустым. Попробуйте снова.")
#         continue
    
#     with torch.no_grad(): 
#         image = pipe(prompt, height=512, width=512, num_inference_steps=1, guidance_scale=0).images[0]
    
#     image_path = get_image_path(prompt)

#     image.save(image_path)
#     print(f"Изображение сохранено в {image_path}")

#     del image
#     gc.collect()




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
    """
    Сохраняет изображение в файл и возвращает путь до файла.
    """
    # Генерация имени файла на основе промпта
    base_name = prompt.replace(" ", "_").strip()
    output_dir = "./generated_images"
    os.makedirs(output_dir, exist_ok=True)  # Создаем директорию, если её нет

    file_path = os.path.join(output_dir, f"{base_name}.png")
    counter = 1

    # Уникализация имени файла
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{base_name}_{counter}.png")
        counter += 1

    image.save(file_path, format="PNG")
    return file_path

def main():
    # Создаём парсер аргументов
    parser = argparse.ArgumentParser(description="Генерация изображений с использованием модели Stable Diffusion.")
    parser.add_argument("model_path", type=str, help="Путь к файлу модели.")  # Позиционный аргумент для модели

    args = parser.parse_args()

    pipe = torch.load(args.model_path, weights_only=False)
    device = "mps"
    pipe.to(device)


    # Читаем ввод из stdin
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:  # Если пустой ввод
            sys.exit(0)
        
        try:
            # Измерение производительности
            start_time = time.time()
            memory_before = psutil.virtual_memory().used

            with torch.no_grad():
                image = pipe(prompt, height=512, width=512, num_inference_steps=1, guidance_scale=0).images[0]
            
            memory_after = psutil.virtual_memory().used
            generation_time = time.time() - start_time

            # Логируем в stderr
            print(f"Генерация завершена: {generation_time:.2f} секунд, "
                  f"использовано памяти: {((memory_after - memory_before) / 1e6):.2f} MB.",
                  file=sys.stderr)

            # Сохраняем изображение в файл
            file_path = save_image_to_file(image, prompt)

            # Выводим путь до файла в stdout
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
