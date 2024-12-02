FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install torch pandas
RUN pip install --no-cache-dir jupyter
EXPOSE 8888
# Set the working directory in the container
WORKDIR /workspace


