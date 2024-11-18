import os
import shutil
import sys
import signal
import time
import platform
import argparse

from python_coreml_stable_diffusion.run import main

from python_coreml_stable_diffusion.coreml_model import (
    CoreMLModel,
    _load_mlpackage,
    _load_mlpackage_controlnet,
    get_available_compute_units,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        default="./coreml-stable-diffusion-2-1-base-palettized_split_einsum_v2_compiled/original/packages",
        help="Path to input directory with the .mlpackage files generated by python_coreml_stable_diffusion.torch2coreml")
    parser.add_argument(
        "-o",
        default="./",
        help="Output directory for generated images")
    parser.add_argument(
        "--seed",
        "-s",
        default=0,
        type=int,
        help="Random seed to be able to reproduce results")
    parser.add_argument(
        "--model-version",
        default="CompVis/stable-diffusion-v1-4",
        help="The pre-trained model checkpoint and configuration to restore.")
    parser.add_argument(
        "--compute-unit",
        choices=get_available_compute_units(),
        default="CPU_AND_NE",
        help="The compute units to be used when executing Core ML models.")
    parser.add_argument(
        "--scheduler",
        default="DDIM",
        help="The scheduler to use for running the reverse diffusion process.")
    parser.add_argument(
        "--num-inference-steps",
        default=20,
        type=int,
        help="The number of iterations the unet model will be executed throughout the reverse diffusion process")
    parser.add_argument(
        "--guidance-scale",
        default=7.5,
        type=float,
        help="Controls the influence of the text prompt on sampling process (0=random images)")
    parser.add_argument(
        "--controlnet",
        nargs="*",
        type=str,
        help="Enables ControlNet and uses control-unet instead of unet for additional inputs.")
    parser.add_argument(
        "--controlnet-inputs",
        nargs="*",
        type=str,
        help="Image paths for ControlNet inputs. Correspond to each controlnet provided at --controlnet option in same order.")
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, pixelated, out of focus, distorted, unnatural, grainy, low resolution, noise, artifacts, smudges, overexposure, underexposure, watermarks, text, logos, faces, people, animals, characters, cartoon, 3D render, frame, borders, gradients, shadows, over-saturated colors, strong reflections, transparent, glossy, neon colors, overly complex shapes",
        help="The negative text prompt to be used for text-to-image generation.")
    parser.add_argument(
        "--unet-batch-one",
        action="store_true",
        help="Do not batch unet predictions for the prompt and negative prompt.")
    parser.add_argument(
        '--model-sources',
        default="compiled",
        choices=['packages', 'compiled'],
        help='Force build from `packages` or `compiled`.')

    args = parser.parse_args()

    while True:
        prompt = input("Введите промпт для генерации изображения ('Ctrl + C' для выхода): ")

        args.prompt = prompt
        
        main(args)
        