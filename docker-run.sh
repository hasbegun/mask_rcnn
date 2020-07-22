#!/usr/bin/env bash
IMG_NAME=alphablocks/mask-rcnn
IMG_VER=latest
CONT_NAME=mask-rcnn

IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
echo "Host IP: $IP"

/usr/X11/bin/xhost +$IP

docker run -it --rm -p 9000:9000 \
    --name $CONT_NAME \
    --net=host \
    -e DISPLAY=$IP:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/home/user/.Xauthority \
    -v $PWD:/home/developer/projects/mrcnn:rw \
    $IMG_NAME:$IMG_VER /bin/bash
