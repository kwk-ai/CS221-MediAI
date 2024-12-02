docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -v "$(pwd)":/workspace -it --rm nvcr.io/nvidia/pytorch:23.08-py3
