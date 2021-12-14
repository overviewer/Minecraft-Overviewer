# syntax=docker/dockerfile:1
FROM ubuntu:20.04

WORKDIR /build

COPY . .

RUN apt-get update
RUN apt-get install -y build-essential python3-pil python3-dev python3-numpy wget

RUN python3 setup.py build

ENV VERSION=1.18.1
RUN mkdir -p ~/.minecraft/versions/$VERSION/
RUN wget https://overviewer.org/textures/$VERSION -O ~/.minecraft/versions/$VERSION/$VERSION.jar

CMD ["/build/overviewer.py", "--config=/server/overviewer.config.py"]
