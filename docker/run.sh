#!/usr/bin/env bash

docker run -it -p 8888:8888 -p 6006:6006 -v ./../:/host deep-learning
