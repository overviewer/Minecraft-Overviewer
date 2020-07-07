FROM python:3
LABEL github="https://github.com/overviewer/Minecraft-Overviewer" maintainer="TheDruidsKeeper"
WORKDIR /usr/local/src

RUN echo "Installing c build tools" \
    && apt-get update \
    && apt-get install build-essential -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Installing python dependencies" \
    && pip3 install --no-cache-dir numpy pillow \
    && echo "Fetching source code for Pillow since it doesn't come with headers" \
    && PILLOW_VERSION=`pip show Pillow | awk '$1 == "Version:" {print $2}'` \
    && wget https://github.com/python-pillow/Pillow/archive/${PILLOW_VERSION}.tar.gz \
    && tar -xf ${PILLOW_VERSION}.tar.gz \
    && rm ${PILLOW_VERSION}.tar.gz \
    && ln -s Pillow-${PILLOW_VERSION} Pillow

# Get & build source code
COPY . minecraft-overviewer
RUN echo "Copying pillow headers" \
    && cp ./Pillow/src/libImaging/ImPlatform.h ./minecraft-overviewer/ImPlatform.h \
    && cp ./Pillow/src/libImaging/Imaging.h ./minecraft-overviewer/Imaging.h \
    && cp ./Pillow/src/libImaging/ImagingUtils.h ./minecraft-overviewer/ImagingUtils.h \
    && echo "Building minecraft-overviewer" \
    && cd minecraft-overviewer \
    && python3 setup.py build

WORKDIR /usr/local/src/minecraft-overviewer
ENTRYPOINT ["python3", "./overviewer.py"]
