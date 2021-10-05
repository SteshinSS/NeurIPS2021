# Конфигурации моделей и датасетов
Мы пишем конфигурации в формате yaml, потому что он простой и понятный. Вот [туториал](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started), вот [Cheat Sheet](https://quickref.me/yaml).

Чтобы прочитать конфиг из Питона выполните:
```python
import yaml
with open('configs/.../my_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```
В `config` будет лежать `dict` с конфигурацией.