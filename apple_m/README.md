# Генерация изображений при помощи Stable Diffusion на Mac с процессорами M

Модели SD были адаптированы при помощи CoreML для работы с NE в процессорах серии M.

## Установка

1. Создайте виртуальное окружение и активируйте его:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```

3. Установите пакет для прямой загрузки с Яндекс.Диска:
    ```bash
    pip install wldhx.yadisk-direct
    ```

4. Скачайте модели:
    ```bash
    curl -L $(yadisk-direct https://disk.yandex.ru/d/VYowq-XDO6crdg
    ) -o coreml-stable-diffusion-2-1-base-palettized_split_einsum_v2_compiled.zip
    ```

5. Распакуйте архив с моделями:
    ```bash
    unzip coreml-stable-diffusion-2-1-base-palettized_split_einsum_v2_compiled.zip -d models
    ```

6. Запустите приложение:
    ```bash
    python image_generator.py
    ```