#!/bin/bash
# Start Nemotron-3-Nano NIM container

export LOCAL_NIM_CACHE=/nim

docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -p 8000:8000 \
    --name nemotron-nano \
    nvcr.io/nim/nvidia/nemotron-3-nano:latest
