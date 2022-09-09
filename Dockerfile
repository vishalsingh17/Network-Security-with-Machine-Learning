FROM python:3.7-slim-buster

COPY . /app

WORKDIR /app

ARG MONGODB_URL

ENV MONGODB_URL $MONGODB_URL

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

CMD [ "python","app.py" ]