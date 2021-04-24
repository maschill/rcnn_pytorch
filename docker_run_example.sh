docker run \
    -it \
    --rm \
    --ipc=host \
    --env="DISPLAY" \
    --runtime=nvidia \
    -v /path/to/rcnn_pytorch:/app \
    pytorch18_1 python /app/run.py

