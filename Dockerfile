FROM ubuntu:18.04

RUN apt-get update
RUN apt-get update && apt-get install -y \
    build-essential \
    python-pil \
    python-dev \
    python-numpy \
    git \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /tmp/overviewer
WORKDIR /tmp/overviewer
ADD https://launcher.mojang.com/v1/objects/30bfe37a8db404db11c7edf02cb5165817afb4d9/client.jar /tmp/overviewer/
RUN chmod 755 /tmp/overviewer/client.jar
RUN ls -la

COPY . /tmp/overviewer
COPY overviewer_core/aux_files/genPOI.py /tmp/overviewer
RUN python2 setup.py build

RUN mkdir /tmp/world
RUN mkdir /tmp/export
RUN mkdir /tmp/config

RUN useradd -ms /bin/bash bob
USER bob

ENTRYPOINT ["/bin/bash", "-c","/tmp/overviewer/overviewer.py --config=/tmp/config/config.py && /tmp/overviewer/overviewer.py --config=/tmp/config/config.py --skip-scan --genpoi"]
