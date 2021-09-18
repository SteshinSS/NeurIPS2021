# KhrameevaLab
**WORK IN PROGRESS**
Авторы соревнования подготовили пайплайн, который хорош для старта, но недостаточен для работы в команде. Этот репозиторий является оберткой над пайплайном, чтобы скрыть технические детали и позволить обучать модели / обрабатывать датасеты не разбираясь в nextflow / docker / viash.

## Установка
### Пайплайн
Авторы соревнования подготовили для нас пайплайн. [Здесь](https://openproblems.bio/neurips_docs/submission/quickstart/) их инструкция, но вкратце:
1) Установите [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-install).
2) Установите [Docker](https://docs.docker.com/get-docker/).
3) Установите [OpenJDK](https://adoptopenjdk.net/?variant=openjdk11&jvmVariant=hotspot). В результате у вас должна работать команда `which java`. Если не работает, добавьте папку bin из установки в [$PATH](https://losst.ru/peremennaya-path-v-linux).

### Наша среда
Мы все используем разные пакеты. Чтобы скрипты работали у всех одинаково, мы используем единую среду.
1) Установите [Conda](https://docs.conda.io/en/latest/miniconda.html).
2) В корне репозитория выполните
```bash
# Create Conda environment
conda create --name nips python=3.8
```
3) Активируйте среду. Это надо делать при каждом открытии терминала.
```bash
conda activate nips
```
4) Установите пакеты
```bash
pip install -r requirements.txt
```
5) (По желанию) Скачайте датасеты
```bash
python download_data.py
```

## Содержание
ToDo

## Содержимое среды
### Как добавить пакет
Можно легко добавить пакеты Питона. Для этого: 
1) Активируйте среду `conda activate nips`.
2) Установите нужный пакет `pip install <package>`.
3) Сохраните новый список пакетов `conda list -e > requirements.txt`.
4) Добавьте новый пакет в README.md.
Если не получается установить новый пакет или вы хотите добавить что-то не являющееся pip-пакетом, напишите в чат.

### Как обновить среду
Если кто-то добавил пакет в среду и обновил requrements.txt, вам надо переустановить локальную среду:
```bash
# Remove current environment
conda deactivate
conda env remove --name nips

# Install new environment
conda create --name nips python=3.8
conda activate nips
pip install -r requirements.txt
```

### Что уже есть
```bash
# Python
python=3.8

# Basic scientific stuff
numpy
scipy
matplotlib
ipython
jupyterlab 
jupyter
pandas
scikit-learn
seaborn

# Deep Learning
torch==1.9.0+cu111
torchvision==0.10.0+cu111
torchaudio==0.9.0
torchtext
einops
hydra-core  # dl configs 
pytorch-lightning
torchmetrics
torchtyping  # typings for pytorch
wandb
pyyaml  # for configs

# For good coding
requests
pytest
pylint  # linter
mypy  # linter
flake8  # linter
yapf  # formatter
autopep8  # formatter

# For their baseline
anndata 
scanpy
```

## Возможные неполадки
Чтобы запускать модельки на GPU, вам нужно настроить вашу ОС. Если вы уже работали с нейросетями локально, то все должно уже работать. Это заметка для биологов, которые захотят запустить скрипт локально.

