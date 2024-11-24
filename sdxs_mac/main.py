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
    output_dir = "./images"
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
    pipe.set_progress_bar_config(disable=True)
    
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
