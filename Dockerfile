FROM python:3.8-alpine
MAINTAINER Pedro Verani

ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8

COPY ./requirements.txt .

RUN apk update && \
	apk upgrade && \
	apk add --no-cache postgresql-dev gcc python3-dev musl-dev build-base linux-headers pcre-dev pcre && \
	pip install --upgrade pip && \
	pip install --require-hashes --no-cache-dir -r requirements.txt && \
	apk del gcc python3-dev musl-dev build-base linux-headers pcre-dev pcre && \
	rm -rf /var/lib/apt/lists/*

COPY . /app
RUN adduser -D tcc && \
	mkdir /app/static && \
	chown -R tcc /app

WORKDIR /app
USER tcc