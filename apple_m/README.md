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
    pip install -r requirements.txt
    ```

4. **Cоздайте папку images**

## Запуск скрипта

```bash
python main_coreml.py -i путь к директроии с .mlpackage
```

### Компиляция

1. **Убедитесь, что `pyinstaller` установлен**:
    ```bash
    pip install pyinstaller
    ```

2. **Выполните команду для компиляции**:

    ```bash
    pyinstaller --onefile main_coreml.py
    ```

    --onefile для того, чтобы сборка была в один файл
    --hidden-import - зависимости