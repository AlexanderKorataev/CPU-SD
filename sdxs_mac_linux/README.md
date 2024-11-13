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
    pip install torch diffusers transformers Pillow numpy nuitka
    ```

3. **Подготовка модели**:
    Поместите файл модели `model.pth` в ту же папку, где находится `main.py`.

4. **Создание папки для изображений**:
    
    ```bash
    mkdir images
    ```

## Запуск скрипта

```bash
python main.py
```

## Компиляция в исполняемый файл

Для создания автономного исполняемого файла можно использовать **Nuitka**. 

### Компиляция

1. **Убедитесь, что `Nuitka` установлена**:
    ```bash
    pip install nuitka
    ```

2. **Выполните команду для компиляции**:
    ```bash
    nuitka --standalone --onefile --lto=no --follow-imports main.py
    ```
