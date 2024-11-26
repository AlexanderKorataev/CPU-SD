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
import warnings
import gc
restore_stdout_stderr()


def output_image_to_stdout(image):
    restore_stdout_stderr()

    from io import BytesIO
    import struct

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    buffer.close()

    string_repr = repr(image_data)

    image_size = len(image_data)

    size_bytes = struct.pack("<I", image_size)

    print(''.join(f'\\x{byte:02x}' for byte in size_bytes))
    print(string_repr[2:-1])

    suppress_stdout_stderr()


def main():
    try:
        suppress_stdout_stderr()

        pipe = StableDiffusionPipeline.from_pretrained('IDKiro/sdxs-512-0.9', weights_only=False)
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
                        image = pipe(prompt, height=512, width=512, num_inference_steps=1, guidance_scale=0).images[0]
                        
                    output_image_to_stdout(image)

                    del image
                    gc.collect()

                except Exception as e:
                    print(f"Ошибка генерации изображения: {e}", file=sys.stderr)
                    sys.exit(1)

            except EOFError:
                # Обрабатываем Ctrl+D (EOF)
                print("Остановка программы (EOF).", file=sys.stderr)
                sys.exit(0)

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
