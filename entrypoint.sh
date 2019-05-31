#!/bin/bash

set -e
if [ "$1" = "run" ]; then
    echo "Running video app.."
    redis-server &
    python video_app.py -s video.mp4 &
    python server.py &

elif [ "$1" = "build" ]; then
    echo "Build.TBD"
fi
