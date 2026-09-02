FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY predict.py .
ENTRYPOINT ["python", "predict.py", "$inputDataset", "$outputDir"]
