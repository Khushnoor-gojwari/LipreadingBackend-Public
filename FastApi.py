
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import os, subprocess, json
import tensorflow as tf
from utils import load_video, load_alignments, num_to_char, load_data
from modelutil import load_model
from time import time
import gdown
import zipfile
# Initialize FastAPI app
app = FastAPI()

# Enable CORS for communication with frontend (e.g., React on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lipreading-front-public-amgp.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load pre-trained lip reading model
model = load_model()
model.load_weights("checkpoints.weights.h5")

# Path where video data is stored
DATA_DIR = "data/s1"

if not os.path.exists(DATA_DIR):
    print(f"{DATA_DIR} not found. Downloading data.zip from Google Drive...")
    # url = 'https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK&confirm=t'
    # output = 'data.zip'
    # gdown.download(url, output, quiet=False)
    gdown.download("https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK", "data.zip", quiet=False)

    print("Extracting data.zip...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete.")

# Endpoint to list all .mpg videos in the dataset folder
@app.get("/videos/")
def list_mpg_videos():
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".mpg")]

# Predict lip-read text from a given video
@app.get("/predict/")
def predict_lip(request: Request, video_name: str = Query(...)):
    try:
        file_base = os.path.splitext(video_name)[0]
        mpg_path = os.path.join(DATA_DIR, video_name)
        mp4_path = os.path.join("temp", f"{file_base}.mp4")
        os.makedirs("temp", exist_ok=True)

        # Convert video from .mpg to .mp4 using ffmpeg
        subprocess.run(["ffmpeg", "-y", "-i", mpg_path, mp4_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Load frames and ground truth alignments
        frames, alignments = load_data(tf.convert_to_tensor(mpg_path))

        # Decode real text
        real_text = tf.strings.reduce_join([num_to_char(char) for char in alignments]).numpy().decode('utf-8')

        # Predict using model and decode output
        yhat = model.predict(tf.expand_dims(frames, axis=0), verbose=0)
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        predicted_text = tf.strings.reduce_join([num_to_char(char) for char in decoded[0]]).numpy().decode('utf-8')

        # Dynamic video URL using request.base_url (not localhost!)
        timestamp = int(time())
        base_url = str(request.base_url).rstrip("/")
        video_url = f"{base_url}/temp/{file_base}.mp4?ts={timestamp}"

        return {
            "real_text": real_text.strip(),
            "predicted_text": predicted_text.strip(),
            "video_url": video_url
        }

    except Exception as e:
        print("❌ Predict Error:", str(e))
# Mount the 'temp' folder to serve video files via URL
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Accuracy file path for caching results
accuracy_file = "accuracy_cache.json"
@app.get("/calculate-accuracy")
def read_cached_accuracy():
    try:
        if os.path.exists(accuracy_file):
            with open(accuracy_file, "r") as f:
                return json.load(f)
        return {
            "error": "Accuracy cache not found. Please generate accuracy_cache.json manually."
        }
    except Exception as e:
        print("❌ Accuracy Error:", str(e))
        return JSONResponse(status_code=500, content={"error": f"Failed to read accuracy: {str(e)}"})



