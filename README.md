# KhrameevaLab
**Работа в прогрессе. Пишите сразу, если что-то у вас не работает.**

Авторы соревнования подготовили пайплайн, который хорош для старта, но недостаточен для работы в команде. Этот репозиторий является оберткой над пайплайном, чтобы скрыть технические детали и позволить обучать модели / обрабатывать датасеты не разбираясь в nextflow / docker / viash.

## Установка
### Пайплайн
Авторы соревнования подготовили для нас пайплайн. Если вы не собираетесь засылать решение на сервер, установите только AWS CLI.

Установка всего пайплайна описана [здесь](https://openproblems.bio/neurips_docs/submission/quickstart/), но вкратце:
1) Установите [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-install).
2) Установите [Docker](https://docs.docker.com/get-docker/).
3) Установите [OpenJDK](https://adoptopenjdk.net/?variant=openjdk11&jvmVariant=hotspot). В результате у вас должна работать команда `which java`. Если не работает, добавьте папку bin из установки в [$PATH](https://losst.ru/peremennaya-path-v-linux).

AWS CLI нужен для скачивания датасетов. Docker и OpenJDK нужны для запуска пайплайна авторов. Они вам нужны только, если вы будете засылать посылку на сервер.

### Наша среда
Мы все используем разные пакеты. Чтобы скрипты работали у всех одинаково, мы используем единую среду.
1) Установите [Conda](https://docs.conda.io/en/latest/miniconda.html).
2) В корне репозитория выполните
```bash
# Create Conda environment
conda create --name nips python=3.8
conda activate nips
conda env config vars set PYTHONPATH=$PWD
conda env config vars set R_LIBS_USER=~/R/nips
conda deactivate
conda activate nips
```
4) Установите пакеты
```bash
pip install -r requirements.txt
```
5) (По желанию) Скачайте официальные датасеты
```bash
python data/download_data.py --official
```
6) Напишите Семену. Он скинет вам ключи доступа от нашего хранилища датасетов. Это позволит вам скачать наши датасеты/чекпоинты и загрузить свои.

7) Если вы используете R, установите нашу R среду. Инструкция [в Notion](https://www.notion.so/R-work-in-progress-b3fd6b10d895483f979306fbff4900b0).

## Содержание
Краткий обзор репозитория. Подробности смотрите в README соответствующих папок.
### `lab_scripts/`
Модели и скрипты, которые используются больше одного раза. 
- Все предполагают запуск или импорт из корня репозитория: `python lab_scripts/data_analysis/my_pipeline.py`.
### `notebooks/`
Исследовательские ноутбуки. Нужны чтобы поделится простым кодом и выводом одновременно. 
- Чтобы запустить или создать ноутбук, запустите jupyter-notebook (или jupyterlab) в корне репозитория. Чтобы сделать импорт скрипта в ячейку пишите путь из корня: `from lab_scripts.models import my_model`.
- Перед коммитом ноутбука подумайте, пригодятся ли еще кому-нибудь ваши функции. Если да, лучше вынести их в `lab_scripts/` и сделать импорт в ноутбук. Это ускорит работу других людей. 
### `data/`
Папка для хранения датасетов и скриптов для их скачки.
- Мы храним датасеты в нашем AWS S3 хранилище. Чтобы пользоваться им, вам надо добавить к себе ключи доступа.
- Папка записана в `.gitignore`, поэтому ее содержимое не загружается в гит. Если хотите добавить скрипт, добавьте в `.gitignore` строчку `!data/my_script.py`.
### `configs/`
Папка для хранения конфигураций.
### `checkpoints/`
Папка для хранения весов моделей.
- Мы храним веса в AWS S3 хранилище. Чтобы пользоваться им, вам надо добавить к себе ключи доступа.
- Папка записана в `.gitignore`, поэтому ее содержимое не загружается в гит. Если хотите добавить скрипт, добавьте в `.gitignore` строчку `!checkpoints/my_script.py`.
### `pipelines/`
Пайплайны авторов соревнования. Нужны для создания посылки.

## Содержимое среды
### Как обновить среду
Если кто-то добавил пакет в среду и обновил requrements.txt, вам надо переустановить локальную среду, чтобы пользоваться новым пакетом. Учтите, что это удалит ваши локальные пакеты.
```bash
# Remove current environment
conda deactivate
conda env remove --name nips

# Install new environment
conda create --name nips python=3.8
conda activate nips
pip install -r requirements.txt
```

### Как добавить пакет
Можно легко добавить пакеты Питона. Для этого: 

0) Обновите свою среду (см. выше).
1) Активируйте среду `conda activate nips`.
2) Установите нужный пакет `pip install <package>`.
3) Сохраните новый список пакетов `pip freeze > requirements.txt`.
4) Добавьте новый пакет в это README.md в раздел *Что уже есть*.

Если не получается установить новый пакет или вы хотите добавить что-то не являющееся pip-пакетом, напишите в чат.

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
torch
torchvision
torchaudio
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
black  # formatter
pre-commit
rpy2  # for R usage from python

# For their baseline
anndata 
scanpy
```
## Рекомендации
### Код-стайл
У нас нет требований к код-стайлу и тестам, но у нас установлен форматтер [black](https://github.com/psf/black) с дефолтными настройками. При каждой попытке коммита, он проверяет код-стайл и исправляет форматтирование. Если он что-то исправил, обновите файлы: 
```bash
git add -u
```
И снова запустите коммит: `git commit -m "My changes"`.

## Возможные неполадки
Чтобы запускать модельки на GPU, вам нужно настроить вашу ОС. Если вы уже работали с нейросетями локально, то все должно уже работать. Это заметка для биологов, которые захотят запустить модельки локально.

