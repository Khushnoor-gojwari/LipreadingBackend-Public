# Base image
FROM tensorflow/tensorflow:2.11.0

# Set working directory
WORKDIR /app

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything in the current directory (where Dockerfile is) into /app
COPY . .

# Expose the port
EXPOSE 10000

# Run FastAPI app
CMD ["uvicorn", "FastApi:app", "--host", "0.0.0.0", "--port", "10000"]
