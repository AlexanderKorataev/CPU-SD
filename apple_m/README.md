# Генерация изображений при помощи SDXS на Mac с процессорами M

Используемая модель https://huggingface.co/apple/coreml-stable-diffusion-2-1-base-palettized

## Запуск

1. **Создайте виртуальное окружение и активируйте его**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Установка зависимостей**:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

    ```bash
    pip install diffusers transformers Pillow numpy
    ```

3. **Подготовка модели**:
    Поместите файл модели `model.pth` в ту же папку, где находится `main.py`.

4. **Cоздайте папку images**:

## Запуск скрипта

```bash
python main.py
```
