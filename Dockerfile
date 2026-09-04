FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY base_model/ base_model/
COPY adapters/ adapters/
COPY predict.py .
ENTRYPOINT ["python3", "predict.py", "$inputDataset", "$outputDir"]
