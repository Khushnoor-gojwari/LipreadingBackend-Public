FROM tensorflow/tensorflow:2.11.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "FastApi:app", "--host", "0.0.0.0", "--port", "10000"]
