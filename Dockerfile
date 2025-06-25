FROM tensorflow/tensorflow:2.11.0

WORKDIR /app

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 10000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "FastApi:app", "--host", "0.0.0.0", "--port", "10000"]
