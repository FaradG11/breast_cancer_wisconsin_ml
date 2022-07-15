Разработка и продуктивизация модели
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

На **пятом** этапе я настроил версионирование моделей. Все готовые модели, метрики, исходные данный и файлы конфигурации хранятся в папке "models/versions/".<br>
<details><summary>Скриншот запуска приложения в консоли</summary><br>

   ![](https://github.com/FaradG11/breast_cancer_wisconsin_ml/blob/main/screenshots/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202022-07-14%20205919.png)
</details>

На **шестом** этапе я создал микросервис REST API на фрэймворке Flask, который, использую полученную модель предсказывает прогноз. Подготовил Docker файл с сервисом. 
Сервис по умолчанию расположен по адресу: http://localhost:5000/

Сервис имеет 2 способа обращения:
1) Запрос к серверу на страницу '/' с телом запроса в виде JSON файла с параметрами обследования;<br>
<details><summary>Скриншот с тестовым запросом</summary><br>

   ![](https://github.com/FaradG11/breast_cancer_wisconsin_ml/blob/main/screenshots/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202022-07-14%20015919.png)
</details>

2) Форма на странице 'predict/' в которую можно вставить JSON с параметрами обследования.<br>
<details><summary>Скриншоты с формой на сайте</summary><br>

![](https://github.com/FaradG11/breast_cancer_wisconsin_ml/blob/main/screenshots/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202022-07-14%20015222.png)
![](https://github.com/FaradG11/breast_cancer_wisconsin_ml/blob/main/screenshots/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202022-07-14%20015123.png)
</details>


Ниже представлена схема проекта:

Project Organization
------------

   
    ├── README.md          <- README с описанием проекта.
    ├── data
    │   └── raw            <- Оригинальные данные для модели проекта.
    │
    ├── docs               <- Sphinx проект
    │ 
    ├── models             <- Папка с дампами последней обученной модели в бинарном формате
    │   │    
    │   ├── versions       <- Папка с сохраненными версиями пердыдущих обучений. В ней хранятся модели в бинарном формате, конфигурации,         
    │   │                     данные и результирующие метрики
    ├── notebooks          <- Оригинальный код, на котором основан данный проект и файл зависимостей для его выполнения в формате Jupyter notebooks.
    │
    ├── references         <- Справочники и другие пояснительные материалы.
    │
    ├── reports            <- Сформированные файлы предсказательной аналитики (HTML, PDF, CSV).
    │
    ├── requirements.txt   <- Файл зависимостей для воспроизведения среды выполнения
    │                         
    ├── requirement_rest_api.txt  <- Файл зависимостей для воспроизведения rest-api сервиса    
    │
    ├── setup.py           <- Сценарий установки модуля проекта.
    │
    ├── ml_breast_cancer          <- Исходный код модуля для использования в данном проекте.
    │   ├── __init__.py    
    │   │
    │   ├── data           <- Код загрузки данных модели.
    │   │
    │   ├── features       <- Код подготовки признаков модели 
    │   │
    │   ├── models         <- Код реализации тренировки модели и прогнозирования.
    │   │
    └── tox.ini            <- Настройки форматирования кода


--------

