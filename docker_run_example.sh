docker run \
    -it \
    --rm \
    --ipc=host \
    --env="DISPLAY" \
    --runtime=nvidia \
    -v /path/to/rcnn_pytorch:/app \
    pytorch17 python /app/run.py

