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
from typing import Optional
from diffusers.models.lora import LoRACompatibleConv
restore_stdout_stderr()

class SeamlessConfig:
    def __init__(self, x_mode='circular', y_mode='circular'):
        self.x_mode = x_mode
        self.y_mode = y_mode
        self.padding_x = (0, 0)
        self.padding_y = (0, 0)
        self.report = []

    def add_report(self, layer, x_mode, y_mode):
        self.report.append(f"Modified layer: {layer}, x_mode={x_mode}, y_mode={y_mode}")

    def save_report(self, filename="modification_report.txt"):
        with open(filename, "w") as f:
            f.write("\n".join(self.report))

def seamless_tiling(pipeline, config: SeamlessConfig):
    def asymmetric_conv2d_convforward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
        self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
        working = torch.nn.functional.pad(input, self.paddingX, mode=config.x_mode)
        working = torch.nn.functional.pad(working, self.paddingY, mode=config.y_mode)
        return torch.nn.functional.conv2d(working, weight, bias, self.stride, torch.nn.modules.utils._pair(0), self.dilation, self.groups)

    targets = [pipeline.vae, pipeline.text_encoder, pipeline.unet]
    convolution_layers = []

    for target in targets:
        for module in target.modules():
            if isinstance(module, torch.nn.Conv2d):
                convolution_layers.append(module)

    for layer in convolution_layers:
        if isinstance(layer, LoRACompatibleConv) and layer.lora_layer is None:
            layer.lora_layer = lambda * x: 0

        layer._conv_forward = asymmetric_conv2d_convforward.__get__(layer, torch.nn.Conv2d)
        config.add_report(layer, config.x_mode, config.y_mode)

    return pipeline

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

def save_image(image, filename):
    """Сохраняет изображение для проверки и отладки"""
    image.save(filename, format="PNG")

def main():
    try:
        suppress_stdout_stderr()

        parser = argparse.ArgumentParser(description="Генерация изображений с использованием модели Stable Diffusion.")
        parser.add_argument("model_path", type=str, help="Путь к файлу модели.")
        parser.add_argument("--x_mode", type=str, default="circular", help="Режим свёртки по X (circular, reflect, constant).")
        parser.add_argument("--y_mode", type=str, default="circular", help="Режим свёртки по Y (circular, reflect, constant).")
        parser.add_argument("--save_report", action="store_true", help="Сохранить отчёт о модифицированных слоях.")
        parser.add_argument("--save_images", action="store_true", help="Сохранять изображения для проверки.")
        args = parser.parse_args()

        # Загружаем пайплайн
        pipe = torch.load(args.model_path, weights_only=False)
        pipe.set_progress_bar_config(disable=True)
        pipe.to("cpu")

        restore_stdout_stderr()

        config = SeamlessConfig(x_mode=args.x_mode, y_mode=args.y_mode)

        while True:
            try:
                line = input()
                prompt = line.strip()
                if not prompt:
                    sys.exit(0)

                if prompt.startswith('@'):
                    modified_pipe = seamless_tiling(pipeline=pipe, config=config)
                    if args.save_report:
                        config.save_report()
                    prompt = prompt[1:].strip()
                    current_pipe = modified_pipe
                else:
                    current_pipe = pipe

                try:
                    with torch.no_grad():
                        image = current_pipe(
                            prompt,
                            height=512,
                            width=512,
                            num_inference_steps=1,
                            guidance_scale=0
                        ).images[0]

                    if args.save_images:
                        save_image(image, f"output_{prompt.replace(' ', '_')}.png")

                    output_image_to_stdout(image)

                    del image
                    gc.collect()

                except Exception as e:
                    sys.stderr.write(f"Error generating image: {e}\n")
                    sys.exit(1)

            except EOFError:
                sys.exit(0)

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
