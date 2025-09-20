from flask import Flask, render_template, request
import os
from utils.frame_extraction import extract_frames
from utils.surf_match import surf_match
from utils.similarity import compare_embeddings
from utils.ocr_text import extract_text
from models.cnn_model import CNNFeatureExtractor
from models.rnn_model import RNNModel
from models.nlp_model import NLPModel

app = Flask(__name__)

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    screenshot = request.files["screenshot"]
    video = request.files["video"]

    screenshot_path = os.path.join(UPLOAD_FOLDER, screenshot.filename)
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)

    screenshot.save(screenshot_path)
    video.save(video_path)

    # Extract video frames
    frames = extract_frames(video_path)

    # CNN embeddings
    cnn = CNNFeatureExtractor()
    screenshot_embedding = cnn.get_embedding(screenshot_path)

    best_match, timestamp = None, None
    best_score = -1

    for idx, frame in enumerate(frames):
        frame_embedding = cnn.get_embedding(frame)
        score = compare_embeddings(screenshot_embedding, frame_embedding)
        if score > best_score:
            best_score = score
            best_match = frame
            timestamp = idx

    return render_template("result.html", score=best_score, timestamp=timestamp)

if __name__ == "__main__":
    app.run(debug=True)
