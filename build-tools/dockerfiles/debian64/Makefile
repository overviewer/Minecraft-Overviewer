tag=debian64

build:
	docker build -t registry.code.em32.net/overviewer/minecraft-overviewer:$(tag) . 

forcebuild:
	docker build --no-cache -t registry.code.em32.net/overviewer/minecraft-overviewer:$(tag) . 

push:
	docker push registry.code.em32.net/overviewer/minecraft-overviewer:$(tag)

run:
	docker run --rm --privileged -it registry.code.em32.net/overviewer/minecraft-overviewer:$(tag) /bin/bash

