#   Данный докер-файл используется для создания контейнера
#   с микросервисом REST-API для определения диагноза пациаента
#   по данным в формате JSON
#   Чтобы запустить контейнер используйте
#   команду 'docker run -it -p 5000:5000 {имя приложения}'

FROM python:3.9-buster

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements_rest_api.txt ./
RUN python -m pip install -r requirements_rest_api.txt

COPY flask-app .
COPY models ./

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "main:app" ]