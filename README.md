# KhrameevaLab
**Некоторые вещи тут -- неправда. Спроси меня, если что-то непонятно, Алин.**

Авторы соревнования подготовили пайплайн, который хорош для старта, но недостаточен для работы в команде. Этот репозиторий является оберткой над пайплайном, чтобы скрыть технические детали и позволить обучать модели / обрабатывать датасеты не разбираясь в nextflow / docker / viash.

## Установка
### Пайплайн
Установите [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-install). Он нужен для скачивания датасетов и чекпоинтов с нашего AWS S3 хранилища.

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
6) (По желанию) Если вы используете R, установите нашу R среду:
    ```bash
    # Create directory for R packages
    mkdir ~/R/nips

    # Install packages
    conda activate nips
    Rscript r_packages.R
    ```
    Инструкция по использованию R находится [тут](https://www.notion.so/R-91166267c5164c2fad426c2e59bd57b7).

7) Напишите Семену. Он скинет вам ключи доступа от нашего хранилища датасетов. Это позволит вам скачать наши датасеты/чекпоинты и загрузить свои.


## Содержание
Краткий обзор репозитория. Подробности смотрите в README соответствующих папок.
### `lab_scripts/`
Модели и скрипты, которые используются больше одного раза. 
- Все предполагают запуск или импорт из корня репозитория: `python lab_scripts/data_analysis/my_pipeline.py`.
### `notebooks/`
Исследовательские ноутбуки. Нужны чтобы поделится простым кодом и выводом одновременно. 
- Чтобы импортировать датасеты по относительному пути, смените рабочую директорию на корень репозитория:
    ```python
    from lab_scripts.utils import utils
    utils.change_directory_to_repo()
    data = ad.read_h5ad("data/raw/gex_adt/azimuth_gex.h5ad")
    ```

- Перед коммитом ноутбука подумайте, пригодятся ли еще кому-нибудь ваши функции. Если да, лучше вынести их в `lab_scripts/` и сделать импорт в ноутбук. Это ускорит работу других людей. 
### `data/`
Папка для хранения датасетов и скриптов для их скачки.
- Мы храним датасеты в нашем AWS S3 хранилище. Чтобы пользоваться им, вам надо добавить к себе ключи доступа.
- Некоторые подпапки записаны в `.gitignore`, поэтому их содержимое не загружается гит. Если хотите добавить скрипт, смотрите `.gitignore`.
### `configs/`
Папка для хранения конфигураций.
### `checkpoints/`
Папка для хранения весов моделей.
- Мы храним веса в AWS S3 хранилище. Чтобы пользоваться им, вам надо добавить к себе ключи доступа.
- Некоторые подпапки записаны в `.gitignore`, поэтому их содержимое не загружается гит. Если хотите добавить скрипт, смотрите `.gitignore`.
### `pipelines/`
Пайплайны авторов соревнования. Нужны для создания посылки.

## Содержимое среды
### Как обновить среду
Если кто-то добавил новый пакет в среду, вам надо установить его, чтобы скрипты работали.
Обновить среду питона:
```bash
conda activate nips
pip install -r requirements.txt
```

Обновить среду R:
```bash
conda activate nips
Rscript r_packages.R
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

# For omics analysis
anndata 
scanpy
louvain  # clusterizing method
```
## Рекомендации
### Код-стайл
У нас нет требований к код-стайлу и тестам, но у нас установлен форматтер [black](https://github.com/psf/black) с дефолтными настройками. Пожалуйста, запустите `black .` в корне репозитория перед коммитом, чтобы он отформатировал файлы. Так же вы можете включить автоматическое форматирование при коммите.
```bash
pre-commit install
```
Теперь при каждой попытке коммита, `black` проверит файлы. Если какие-то файлы неотформатированны, он их поправит и отменит коммит. Обновите отформатированные файлы и повторите коммит:
```bash
git add -u
git commit -m ...
```

## Возможные неполадки
Чтобы запускать модельки на GPU, вам нужно настроить вашу ОС. Если вы уже работали с нейросетями локально, то все должно уже работать. Это заметка для биологов, которые захотят запустить модельки локально.

