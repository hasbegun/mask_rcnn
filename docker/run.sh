#!/usr/bin/env bash
IMG_NAME=mrcnn

docker run -it -p 8888:8888 -p 6006:6006 -v ./../:/host $IMG_NAME
