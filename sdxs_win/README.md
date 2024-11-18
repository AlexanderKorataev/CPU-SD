# Генерация изображений при помощи SDXS на Windows

Используемая модель [IDKiro/sdxs-512-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9)

## Запуск

1. **Создайте виртуальное окружение и активируйте его**:
    Откройте командную строку (или PowerShell) в директории проекта и выполните:

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2. **Установка зависимостей**:

    ```bash
    pip install torch --index-url https://download. pytorch.org/whl/cpu
    ```

    ```bash
    pip install diffusers transformers Pillow numpy pyinstaller
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
    pyinstaller --onefile --hidden-import=transformers --hidden-import=torch --hidden-import=diffusers --hidden-import=safetensors main.py
    ```