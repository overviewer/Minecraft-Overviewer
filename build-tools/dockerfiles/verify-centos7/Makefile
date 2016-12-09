tag=centos7-verify

build:
	docker build -t code.em32.net:14567/overviewer/minecraft-overviewer:$(tag) . 

push:
	docker push code.em32.net:14567/overviewer/minecraft-overviewer:$(tag)

run:
	docker run --rm -it code.em32.net:14567/overviewer/minecraft-overviewer:$(tag) /bin/bash

