# Сохраненные веса моделей.
Веса моделей хранятся в нашем AWS S3 хранилище. Чтобы скачать / загрузить веса, получите ключи доступа от Семена.

## Как скачать чекпоинты
```bash
python checkpoints/download_checkpoints.py
```

## Как сохранить чекпоинты в хранилище
```bash
# Upload folder
aws s3 sync checkpoints/my_model s3://nips2021/checkpoints/my_model

# Upload checkpoint
aws s3 cp checkpoints/my_model/model.ckpt s3://nips2021/checkpoints/my_model/model.ckpt
```

Подробности смотрите [тут](https://www.notion.so/f4f99add031b4c4ab06bd443a732c811)
