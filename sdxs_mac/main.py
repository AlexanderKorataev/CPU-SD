import sys
import os


def suppress_stdout_stderr():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def restore_stdout_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


suppress_stdout_stderr()
import torch
from diffusers import StableDiffusionPipeline
import struct
from io import BytesIO
from PIL import Image
import gc
import argparse
restore_stdout_stderr()


def output_image_to_stdout(image):
    restore_stdout_stderr()

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    buffer.close()

    image_size = len(image_data)

    size_bytes = struct.pack("<I", image_size)

    sys.stderr.write(f"{image_size} \n")
    sys.stdout.buffer.write(size_bytes)
    sys.stdout.buffer.flush()

    sys.stdout.buffer.write(image_data)
    sys.stdout.buffer.flush()

    suppress_stdout_stderr()


def main():
    try:
        suppress_stdout_stderr()

        parser = argparse.ArgumentParser(description="Генерация изображений с использованием модели Stable Diffusion.")
        parser.add_argument("model_path", type=str, help="Путь к файлу модели.")
        args = parser.parse_args()

        # pipe = StableDiffusionPipeline.from_pretrained('IDKiro/sdxs-512-0.9', weights_only=False)
        pipe = torch.load(args.model_path, weights_only=False)
        pipe.set_progress_bar_config(disable=True)
        pipe.to("mps")
        restore_stdout_stderr()

        while True:
            try:
                line = input()
                prompt = line.strip()
                if not prompt:
                    sys.exit(0)

                try:
                    with torch.no_grad():
                        image = pipe(
                            prompt,
                            height=512,
                            width=512,
                            num_inference_steps=1,
                            guidance_scale=0
                        ).images[0]

                    output_image_to_stdout(image)

                    del image
                    gc.collect()

                except Exception as e:
                    sys.exit(1)

            except EOFError:
                sys.exit(0)

    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
