import cv2
import tensorflow as tf
import numpy as np
from modelutil import load_model
from utils import num_to_char, load_video
import editdistance
import sys

# Load the trained model
model = load_model()
model.load_weights('model/checkpoint').expect_partial()

# Decode prediction into text
def decode_prediction(pred):
    decoded = tf.keras.backend.ctc_decode(pred, input_length=[75], greedy=True)[0][0].numpy()[0]
    return tf.strings.reduce_join([num_to_char(char) for char in decoded]).numpy().decode('utf-8')

# Accuracy calculation
def compute_accuracy(true_text, pred_text):
    char_dist = editdistance.eval(true_text, pred_text)
    char_acc = (1 - char_dist / max(len(true_text), 1)) * 100

    real_words = true_text.split()
    pred_words = pred_text.split()
    word_dist = editdistance.eval(real_words, pred_words)
    word_acc = (1 - word_dist / max(len(real_words), 1)) * 100

    return char_acc, word_acc

# Process downloaded video
def test_on_downloaded_video(video_path, real_text=None):
    frames = load_video(video_path)  # shape: (num_frames, 46, 140, 1)
    
    # Ensure frame count is exactly 75
    max_len = 75
    num_frames = frames.shape[0]
    
    if num_frames > max_len:
        frames = frames[:max_len]
    elif num_frames < max_len:
        padding = np.zeros((max_len - num_frames, 46, 140, 1), dtype=np.float32)
        frames = np.concatenate([frames, padding], axis=0)

    # Expand dimensions for batch input
    prediction = model.predict(tf.expand_dims(frames, axis=0))
    pred_text = decode_prediction(prediction)

    print("=" * 80)
    print("PREDICTED TEXT:", pred_text)

    if real_text:
        char_acc, word_acc = compute_accuracy(real_text.lower(), pred_text.lower())
        print(f"\nCharacter Accuracy: {char_acc:.2f}%")
        print(f"Word Accuracy: {word_acc:.2f}%")
    else:
        print("No ground truth provided — skipping accuracy computation.")
    print("=" * 80)


# Capture real-time video (5 seconds max, or press 'q' to quit early)
def record_realtime_video(filename='realtime_test.avi', duration=5):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 25.0, (160, 120))

    print(f"Recording {duration} seconds from webcam. Press 'q' to stop early.")
    frame_count = 0
    max_frames = duration * 25

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (160, 120))
        out.write(frame)
        frame_count += 1

        cv2.imshow('Recording (Press q to stop)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Recording saved.")
    return filename

# Main
if __name__ == '__main__':
    mode = input("Select mode: (1) Downloaded Video (2) Real-Time Webcam\nEnter 1 or 2: ")

    if mode == '1':
        video_path = input("Enter path to downloaded video file: ").strip()
        real_text = input("Enter the actual ground truth sentence (press Enter to skip): ").strip()
        test_on_downloaded_video(video_path, real_text if real_text else None)

    elif mode == '2':
        print("Starting real-time recording...")
        filename = record_realtime_video()
        real_text = input("Enter the actual ground truth sentence: ").strip()
        test_on_downloaded_video(filename, real_text)

    else:
        print("Invalid selection.")
