# Сохраненные веса моделей.
Веса моделей хранятся в нашем AWS S3 хранилище. Чтобы скачать / загрузить веса, получите ключи доступа от Семена.

## Как скачать чекпоинты
```bash
python checkpoints/download_checkpoints.py
```

## Как сохранить чекпоинты в хранилище
```bash
aws s3 cp checkpoints/my_checkpoint s3://nips2021/checkpoints/my_checkpoint
```

## Как обновить измененные чекпоинты
```
aws s3 sync checkpoints s3://nips2021/checkpoints
```

Подробности смотрите [тут](https://www.notion.so/f4f99add031b4c4ab06bd443a732c811)


## Что за `init` файлы
Гит не разрешает пушить пустые папки, поэтому я кладу в них пустой файлик `init`.