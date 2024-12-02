docker run -it --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864  -v "$(pwd)":/workspace -it --rm kvasir_demo:latest
