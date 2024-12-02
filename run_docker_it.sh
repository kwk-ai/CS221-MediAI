docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "$(pwd)":/workspace -w /workspace --network=host -it cs221_healthcare_project /bin/bash 
