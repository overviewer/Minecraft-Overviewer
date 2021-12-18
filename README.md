
# Minecraft Overviewer Docker

This reposetory is a fork of the [Minecraft Overviewer repository](https://github.com/overviewer/Minecraft-Overviewer). This is intended to serve as an example of how to build and use this project as a Docker image. To build the docker image, run `./build-overviewer.sh` and then run `./run-overviewer.sh` to run the resulting docker image.

## Documentation & GitHub

The Minecraft Overviewer documentation can be found [here](http://docs.overviewer.org/). The official GitHub page for the project can be foud [here](https://github.com/overviewer/Minecraft-Overviewer).

## Usage

This section outlines basic usage of the docker images for Minecraft Overviewer. The examples given will assume that you have a minecraft server installed and that your current working directory is the server directory. If, for instance, you've installed your server to `~/server` it is assumed that this is your current working directory.

## Core

Overviewer depends on textures in order to render your world. These can come either from a client `.jar` or a resource pack. See the [documentation](http://docs.overviewer.org/en/latest/running/#installing-the-textures) for more detail instructions on installing textures.

For this example, we will assume you're downloading the 1.18.1 client `.jar` distributed on the Overviewer website.

```console
user@desktop:~/server$ mkdir overviewer
user@desktop:~/server$ wget https://overviewer.org/textures/1.18.1 -O ./overviewer/1.18.1.jar
```

Next, add download the sample config file to `~/server/overviewer/config.py`.

```console
user@desktop:~/server$ wget https://raw.githubusercontent.com/bcheidemann/Minecraft-Overviewer-Docker/master/sample_config.py -O ./overviewer/config.py
```

Finally, run the docker image. The map will be rendered to `~/server/overviewer/output`.

```console
user@desktop:~/server$ docker run \
  --name overviewer-test \
  -v $PWD/world:/world \
  -v $PWD/overviewer/output:/output \
  -v $PWD/overviewer/config.py:/overviewer/config.py \
  -v $PWD/overviewer/1.18.1.jar:/root/.minecraft/versions/1.18.1/1.18.1.jar \
  overviewer --config=/overviewer/config.py
```

You can then view the render by opening `~/server/overviewer/output/index.html` in a browser.
