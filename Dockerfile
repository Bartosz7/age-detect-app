# syntax=docker/dockerfile:1
FROM python:3.11
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /age-detection/src
COPY ./requirements.txt /age-detection
WORKDIR /age-detection/src

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install vim nano -y

RUN pip install --upgrade pip
RUN pip install -r ../requirements.txt
