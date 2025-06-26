# from fastapi import FastAPI, Query, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import JSONResponse
# import os, subprocess, json, zipfile
# import tensorflow as tf
# from utils import load_video, load_alignments, num_to_char, load_data
# from modelutil import load_model
# from time import time
# import gdown

# app = FastAPI(redirect_slashes=False)

# # ✅ CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://lipreading-front-public-amgp.vercel.app"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Load model
# model = load_model()
# model.load_weights("checkpoints.weights.h5")

# # ✅ Use local data (skip download if not needed)
# DATA_DIR = "data/s1"
# if not os.path.exists(DATA_DIR):
#     print(f"{DATA_DIR} not found. Downloading...")
#     gdown.download("https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK", "data.zip", quiet=False)
#     with zipfile.ZipFile("data.zip", 'r') as zip_ref:
#         zip_ref.extractall(".")
#     print("Extraction complete.")

# # ✅ List videos
# @app.get("/videos/")
# def list_mpg_videos():
#     return [f for f in os.listdir(DATA_DIR) if f.endswith(".mpg")]

# # ✅ Predict route
# @app.get("/predict")
# @app.get("/predict/")
# def predict_lip(request: Request, video_name: str = Query(...)):
#     try:
#         file_base = os.path.splitext(video_name)[0]
#         mpg_path = os.path.join(DATA_DIR, video_name)
#         mp4_path = os.path.join("temp", f"{file_base}.mp4")
#         os.makedirs("temp", exist_ok=True)

#         # subprocess.run(["ffmpeg", "-y", "-i", mpg_path, mp4_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         result = subprocess.run(["ffmpeg", "-y", "-i", mpg_path, mp4_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#         if result.returncode != 0:
#             error_message = result.stderr.decode()
#             print("❌ ffmpeg failed:\n", error_message)
#             return JSONResponse(status_code=500, content={"error": "ffmpeg failed", "details": error_message})
#         frames, alignments = load_data(tf.convert_to_tensor(mpg_path))

#         real_text = tf.strings.reduce_join([num_to_char(char) for char in alignments]).numpy().decode('utf-8')

#         yhat = model.predict(tf.expand_dims(frames, axis=0), verbose=0)
#         decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
#         predicted_text = tf.strings.reduce_join([num_to_char(char) for char in decoded[0]]).numpy().decode('utf-8')

#         # 🔧 Use dynamic host instead of localhost
#         timestamp = int(time())
#         base_url = str(request.base_url).rstrip("/")
#         video_url = f"{base_url}/temp/{file_base}.mp4?ts={timestamp}"

#         return {
#             "real_text": real_text.strip(),
#             "predicted_text": predicted_text.strip(),
#             "video_url": video_url
#         }

#     except Exception as e:
#         print("❌ Predict Error:", str(e))
#         return JSONResponse(status_code=500, content={"error": str(e)})
        

# # ✅ Mount static video files
# app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# # ✅ Accuracy endpoint
# accuracy_file = "accuracy_cache.json"

# @app.get("/calculate-accuracy")
# def read_cached_accuracy():
#     try:
#         if os.path.exists(accuracy_file):
#             with open(accuracy_file, "r") as f:
#                 return json.load(f)
#         return {
#             "error": "Accuracy cache not found. Please generate accuracy_cache.json manually."
#         }
#     except Exception as e:
#         print("❌ Accuracy Error:", str(e))
#         return JSONResponse(status_code=500, content={"error": f"Failed to read accuracy: {str(e)}"})


from fastapi import FastAPI , Query ,UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import subprocess
import tempfile
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
from time import time
import json  # Import json module

app = FastAPI(redirect_slashes=False)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://lipreading-front-public-amgp.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = load_model()
model.load_weights("checkpoints.weights.h5")

@app.get("/")
def health():
    return {"status": "Lipreading backend is alive"}

@app.get("/videos/")
def list_mpg_videos():
    # List available .mpg videos
    return [f for f in os.listdir("data/s1") if f.endswith(".mpg")]

# @app.post("/predict")
# async def predict_lip(file: UploadFile = File(...)):
#     if not file:
#         return JSONResponse(status_code=400, content={"error": "File is required"})

#     video_bytes = await file.read()

#     # Create a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
#         temp_video_path = temp_video_file.name
#         temp_video_file.write(video_bytes)

#     # Process the video using ffmpeg
#     output_video_path = temp_video_path.replace(".mp4", "_processed.mp4")
#     result = subprocess.run(
#         ["ffmpeg", "-y", "-i", temp_video_path, output_video_path],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )
    
#     if result.returncode != 0:
#         err = result.stderr.decode()
#         return JSONResponse(status_code=500, content={"error": "ffmpeg failed", "details": err})

#     # Load & run your model
#     try:
#         frames, alignments = load_data(tf.convert_to_tensor(output_video_path))
#         real_text = tf.strings.reduce_join([num_to_char(c) for c in alignments]).numpy().decode("utf-8")
#         yhat = model.predict(tf.expand_dims(frames, axis=0), verbose=0)
#         decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
#         predicted_text = tf.strings.reduce_join([num_to_char(c) for c in decoded[0]]).numpy().decode("utf-8")
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": "model inference failed", "details": str(e)})
@app.post("/predict")
def predict_lip(video_name: str = Query(...)):
    print(f"✅ /predict called with video_name: {video_name}")

    video_path = os.path.join("data/s1", video_name)
    print(f"🔍 Full path: {video_path}")

    if not os.path.exists(video_path):
        print("❌ File not found.")
        return JSONResponse(status_code=404, content={"error": f"{video_name} not found on server."})

    try:
        frames, alignments = load_data(tf.convert_to_tensor(video_path))
        print("✅ Loaded data successfully.")

        real_text = tf.strings.reduce_join([num_to_char(c) for c in alignments]).numpy().decode("utf-8")
        yhat = model.predict(tf.expand_dims(frames, axis=0), verbose=0)
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        predicted_text = tf.strings.reduce_join([num_to_char(c) for c in decoded[0]]).numpy().decode("utf-8")

        print(f"✅ Prediction done: Real: {real_text.strip()}, Predicted: {predicted_text.strip()}")

        return {
            "real_text": real_text.strip(),
            "predicted_text": predicted_text.strip()
        }

    except Exception as e:
        print(f"❌ Inference error: {e}")
        return JSONResponse(status_code=500, content={"error": "Model inference failed", "details": str(e)})        



  

# @app.get("/video-file/{filename}")
# def serve_temp_video(filename: str):
#     file_path = os.path.join("temp", filename)
#     if not os.path.exists(file_path):
#         return JSONResponse(status_code=404, content={"error": "Video file not found."})
#     return FileResponse(file_path, media_type="video/mp4")

@app.get("/calculate-accuracy")
def read_cached_accuracy():
    try:
        accuracy_file = "accuracy_cache.json"
        if os.path.exists(accuracy_file):
            with open(accuracy_file, "r") as f:
                return json.load(f)
        return {"error": "Accuracy cache not found."}
    except Exception as e:
        print("❌ Accuracy Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

