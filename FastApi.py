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
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os, subprocess, json, zipfile
import tensorflow as tf
from utils import load_video, load_alignments, num_to_char, load_data
from modelutil import load_model
from time import time
import gdown

app = FastAPI(redirect_slashes=False)

# ✅ Add CORS middleware early
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lipreading-front-public-amgp.vercel.app"],["*"]  # Your Vercel frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
model = load_model()
model.load_weights("checkpoints.weights.h5")

# ✅ Ensure temp folder exists
os.makedirs("temp", exist_ok=True)

# ✅ Use local data
DATA_DIR = "data/s1"
if not os.path.exists(DATA_DIR):
    print(f"{DATA_DIR} not found. Downloading...")
    gdown.download("https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK", "data.zip", quiet=False)
    with zipfile.ZipFile("data.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete.")

# ✅ List available .mpg videos
@app.get("/videos/")
def list_mpg_videos():
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".mpg")]

# ✅ Predict route
@app.get("/predict")
def predict_lip(request: Request, video_name: str = Query(...)):
    # 1️⃣ Ensure we actually got a video_name
    if not video_name:
        return JSONResponse(status_code=400, content={"error": "query parameter video_name is required"})

    # 2️⃣ Build and validate the path
    mpg_path = os.path.join(DATA_DIR, video_name)
    if not os.path.isfile(mpg_path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Video file not found: {video_name}"}
        )

    mp4_path = os.path.join("temp", f"{os.path.splitext(video_name)[0]}.mp4")
    os.makedirs("temp", exist_ok=True)

    # now you can safely call ffmpeg…
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", mpg_path, mp4_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        err = result.stderr.decode()
        print("❌ ffmpeg failed for", mpg_path, "\n", err)
        return JSONResponse(status_code=500, content={"error":"ffmpeg failed","details":err})

    # …rest of your code…

# ✅ Serve video file through route
@app.get("/video-file/{filename}")
def serve_temp_video(filename: str):
    file_path = os.path.join("temp", filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Video file not found."})
    return FileResponse(file_path, media_type="video/mp4")

# ✅ Accuracy cache endpoint
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
