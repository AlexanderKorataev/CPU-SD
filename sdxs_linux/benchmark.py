import argparse
import torch
from diffusers import StableDiffusionPipeline
import os
import time
import psutil
import pandas as pd
from PIL import Image
import gc
import sys


def load_model(model_path, device="cpu"):
    """
    Загружает модель из указанного пути.
    """
    print(f"Загрузка модели из {model_path}...", file=sys.stderr)
    pipe = torch.load(model_path, weights_only=False)
    pipe.to(device)
    return pipe


def save_image_to_file(image, prompt):
    """
    Сохраняет изображение в файл и возвращает путь до файла.
    """
    base_name = prompt.replace(" ", "_").strip()
    output_dir = "./generated_images"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{base_name}.png")
    counter = 1

    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{base_name}_{counter}.png")
        counter += 1

    image.save(file_path, format="PNG")
    return file_path


def get_total_memory_usage():
    """
    Возвращает общую используемую память (RAM + SWAP) в MB.
    """
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return (vm.used + swap.used) / 1e6  # Память в MB




def benchmark(pipe, prompts, device="cpu"):
    """
    Выполняет бенчмарк на наборе промптов и возвращает результаты в виде таблицы.
    """
    # Измерение общей памяти до запуска модели
    initial_memory = get_total_memory_usage()
    print(f"Общая память до загрузки модели: {initial_memory:.2f} MB", file=sys.stderr)

    results = []

    for prompt in prompts:
        try:
            # Измерение памяти перед генерацией
            memory_before = get_total_memory_usage()
            start_time = time.time()

            with torch.no_grad():
                image = pipe(prompt, height=512, width=512, num_inference_steps=1, guidance_scale=0).images[0]

            # Измерение времени и памяти после генерации
            generation_time = time.time() - start_time
            memory_after = get_total_memory_usage()
            memory_usage = memory_after - memory_before

            # Сохранение изображения
            save_image_to_file(image, prompt)

            # Запись результатов
            results.append({
                "Prompt": prompt,
                "Generation Time (s)": round(generation_time, 2),
                "Memory Usage (MB)": round(memory_usage, 2),
                "Iterations/s": round(50 / generation_time, 2) if generation_time > 0 else 0
            })

            del image
            gc.collect()

        except Exception as e:
            print(f"Ошибка при обработке промпта '{prompt}': {e}", file=sys.stderr)
            results.append({
                "Prompt": prompt,
                "Generation Time (s)": "Error",
                "Memory Usage (MB)": "Error",
                "Iterations/s": "Error"
            })

    # Возвращаем результаты как DataFrame
    return pd.DataFrame(results)



def main():
    # Парсер аргументов
    parser = argparse.ArgumentParser(description="Бенчмарк модели Stable Diffusion.")
    parser.add_argument("model_path", type=str, help="Путь к файлу модели.")
    parser.add_argument("--output", type=str, default="benchmark_results.xlsx", help="Имя файла для сохранения результатов.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Устройство для генерации (cpu, cuda, mps).")

    args = parser.parse_args()

    # Набор промптов
    prompts = [
        "sunset over mountains",
        "a futuristic cityscape",
        "a cat sitting on a tree",
        "tropical beach during sunset",
        "portrait of a king",
        "a medieval castle surrounded by fog",
        "a spaceship in deep space",
        "a robot playing chess",
        "a vibrant jungle scene",
        "a snowy mountain range",
        "a desert with a caravan of camels",
        "a bustling city street at night",
        "a serene lake surrounded by trees",
        "a modern skyscraper skyline",
        "a fantasy forest with glowing trees",
        "an ancient temple in ruins",
        "a magical wizard casting a spell",
        "a pirate ship on the high seas",
        "a cyberpunk-style neon-lit alley",
        "a peaceful village in the countryside"
    ]

    # Загрузка модели
    pipe = load_model(args.model_path, device=args.device)

    # Запуск бенчмарка
    results = benchmark(pipe, prompts)

    # Сохранение результатов в файл
    results.to_excel(args.output, index=False)
    print(f"Результаты бенчмарка сохранены в файл {args.output}")


if __name__ == "__main__":
    main()
