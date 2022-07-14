ML project
==============================

Целью данного проекта являлось создание модели машинного обучения для определения типа опухоли молочной железы используя метод К ближайших соседей(kNN)

На **первом** этапе, используя jupyter notebook, я провел следующие исследования:
    - анализ распределения признаков, в зависимости от типа опухоли,
    - анализ на коррелиарность признаков
на основе исследований я отобрал наиболее подходящие признаки.

На **втором** этапе, я провел подбор гиперпараметров для алгоритма kNN методом кросс-валидации. А также сравнил результаты с результатами работы алгоритма Логистической регрессии.
По итогу я построил модель с оптимальными показателями. Ноутбук с исследованиями лежит в папке "/notebooks"

На **третьем** этапе полученный алгоритм я перевел в структурированный python проект. В качестве шаблона я использовал "ml cookiecutter template". Параметры модели записываются в файл конфигурации в папке "configs".
Ниже инструкция по установке и запуску скрипта. 

Установка: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Запуск:
~~~
python ml_breast_cancer/train_pipeline.py configs/train_config.yaml
~~~

На **четвертом** этапе при помощи инструментов pytest я написал тесты, покрывающие основные этапы пайплайна. 

Чтобы запустить тесты воспользуйтесь командой:
~~~
pytest tests/
~~~

На **пятом** этапе я настроил версионирование моделей. Все готовые модели, метрики, исходные данный и файлы конфигурации хранятся в папке "models/versions/".

   ![](screenshots/Снимок экрана 2022-07-14 205919.png)

На **шестом** этапе я создал микросервис REST API на фрэймворке Flask, который, использую полученную модель предсказывает прогноз. Подготовил Docker файл с сервисом. 

Сервис имеет 2 способа обращения:
1) Запрос к серверу на страницу '/' с телом запроса в виде JSON файла с параметрами обследования;
   ![](screenshots/Снимок экрана 2022-07-14 015919.png)

2) Форма на странице 'predict/' в которую можно вставить JSON с параметрами обследования.
   ![](screenshots/Снимок экрана 2022-07-14 015123.png) ![](screenshots/Снимок экрана 2022-07-14 015222.png)


Ниже представлена схема проекта:

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_example                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

