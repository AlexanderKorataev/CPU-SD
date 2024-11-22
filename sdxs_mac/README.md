# Генерация изображений при помощи SDXS на Win

Используемая модель https://huggingface.co/IDKiro/sdxs-512-0.9


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

## Компиляция в исполняемый файл

Для создания автономного исполняемого файла можно использовать **pyinstaller**. 

### Компиляция

1. **Убедитесь, что `pyinstaller` установлен**:
    ```bash
    pip install pyinstaller
    ```

2. **Выполните команду для компиляции**:

    ```bash
    pyinstaller --hidden-import=torch --hidden-import=diffusers --hidden-import=transformers --hidden-import=huggingface_hub --hidden-import=tokenizers --hidden-import=networkx --hidden-import=safetensors --hidden-import=regex --hidden-import=numpy --hidden-import=numpy.core.multiarray --hidden-import=numpy.core._dtype --hidden-import=requests --onefile
    ```

    --onefile для того, чтобы сборка была в один файл
    --hidden-import - зависимости