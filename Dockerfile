FROM python:3.8
MAINTAINER Pedro Verani

ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8

COPY ./requirements.txt .

RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN adduser tcc && \
	mkdir /app/static && \
	chown -R tcc /app

EXPOSE 5000
WORKDIR /app
USER tcc