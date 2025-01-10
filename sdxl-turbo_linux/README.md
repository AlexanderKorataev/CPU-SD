# Генерация изображений при помощи SDXL-turbo на Linux

Используемая модель https://huggingface.co/stabilityai/sdxl-turbo

## Установите Python

```bash
sudo apt update
sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 -y
sudo apt install python3.12-full
```

*установка через пакетный менеджер apt (Подходит для ubuntu, debian). Для других дистрибутивов используйте соответсвенно их пакетный манеджер

## Запуск

1. **Создайте виртуальное окружение и активируйте его, а также обновите pip**:

    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
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
    pyinstaller --hidden-import=torch --hidden-import=diffusers --hidden-import=transformers --hidden-import=huggingface_hub --hidden-import=tokenizers --hidden-import=networkx --hidden-import=safetensors --hidden-import=regex --hidden-import=numpy --hidden-import=numpy.core.multiarray --hidden-import=numpy.core._dtype --hidden-import=requests --onefile main.py
    ```

    --onefile для того, чтобы сборка была в один файл
    --hidden-import - зависимости