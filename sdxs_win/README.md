# Генерация изображений при помощи SDXS на Windows

Используемая модель [IDKiro/sdxs-512-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9)

## Установите Python

По этой ссылке выберите версию Python, которая вас интересует: https://www.python.org/downloads/release/python-3126/

например Windows installer (64-bit)

## Запуск

1. **Создайте виртуальное окружение и активируйте его, а также обновите pip**:
    Откройте командную строку (или PowerShell) в директории проекта и выполните:

    ```bash
    python3.12 -m venv venv
    .\venv\Scripts\activate
    pip install -U pip
    ```

2. **Установка зависимостей**:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

    ```bash
    pip install diffusers transformers Pillow numpy
    ```

3. **Сохранение модели**

    Запустите save_model.py

    ```bash
    python save_model.py
    ```

4. **Подготовка модели**:
    Убедитесь, что файл модели `model.pth` находится в той же папке, где и `main.py`.

5. **Cоздайте папку images**:

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
