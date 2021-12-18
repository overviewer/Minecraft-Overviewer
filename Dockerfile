# syntax=docker/dockerfile:1
FROM ubuntu:20.04

WORKDIR /build

COPY . .

RUN apt-get update
RUN apt-get install -y build-essential python3-pil python3-dev python3-numpy wget

RUN python3 setup.py build

RUN python3 /build/overviewer.py --version

ENTRYPOINT ["/build/overviewer.py"]
