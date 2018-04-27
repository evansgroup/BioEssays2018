FROM python:3

MAINTAINER German Gonzalez <ggonzale@sierra-research.com>

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

