# Base Image
FROM allennlp/allennlp

RUN apt-get update && apt-get install -y vim\
    emacs\
    zip\
    unzip\
    wget

WORKDIR /nlp4dh

COPY ./lib /nlp4dh/lib
COPY ./models /nlp4dh/models

ENV PYTHONPATH /stage/allennlp/:$PATH

RUN mkdir scripts
RUN mkdir data
