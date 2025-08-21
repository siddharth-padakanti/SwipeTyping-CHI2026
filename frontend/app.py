import os
import math
import torch
from flask import Flask, request, jsonify, render_template, url_for, Blueprint
from flask_cors import CORS
import json
from json.decoder import JSONDecodeError
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
from datetime import datetime
import requests

# === APP SETUP ===

# Backend API URL - change this based on your backend server location
BACKEND_URL = "http://ramen.usask.ca:5000"

typingPage = Blueprint('typing', __name__, template_folder='templates', static_folder='static')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Ensure a folder for participant logs
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Path for our JSON registry
PARTICIPANTS_FILE = os.path.join(LOGS_DIR, 'participants.json')
if not os.path.exists(PARTICIPANTS_FILE):
    with open(PARTICIPANTS_FILE, 'w') as f:
        json.dump([], f)

# Normalized keyboard layout in range (-1,-1) to (1,1)
keyboard_layout = {
    "Q": (-0.9, -0.9), "W": (-0.7, -0.9), "E": (-0.5, -0.9), "R": (-0.3, -0.9), "T": (-0.1, -0.9),
    "Y": (0.1, -0.9), "U": (0.3, -0.9), "I": (0.5, -0.9), "O": (0.7, -0.9), "P": (0.9, -0.9),
    "A": (-0.8, -0.3), "S": (-0.6, -0.3), "D": (-0.4, -0.3), "F": (-0.2, -0.3), "G": (0.0, -0.3),
    "H": (0.2, -0.3), "J": (0.4, -0.3), "K": (0.6, -0.3), "L": (0.8, -0.3),
    "Z": (-0.6, 0.3), "X": (-0.4, 0.3), "C": (-0.2, 0.3), "V": (0.0, 0.3), "B": (0.2, 0.3),
    "N": (0.4, 0.3), "M": (0.6, 0.3)
}

# def angle_to_chars(char, angle):
#     char = char.upper()
#     if char not in keyboard_layout:
#         return []

#     cx, cy = keyboard_layout[char]
#     angle = (angle + 360) % 360
#     radians = math.radians(angle)
#     dx = math.cos(radians)
#     dy = math.sin(radians)

#     result = [char.lower()] * 3
#     visited = set(result)

#     for step in range(1, 15):  # allow longer reach
#         fx = cx + dx * step * 0.2
#         fy = cy + dy * step * 0.2

#         closest = None
#         min_dist = float("inf")

#         for k, (kx, ky) in keyboard_layout.items():
#             if k.lower() in visited:
#                 continue
#             dist = (fx - kx)**2 + (fy - ky)**2
#             if dist < 0.2 and dist < min_dist:  # wider radius
#                 closest = k.lower()
#                 min_dist = dist

#         if closest:
#             visited.add(closest)
#             multiplier = 5 if closest in "aeiou" else 2
#             result.extend([closest] * multiplier)
#             if len(visited) >= 5:
#                 break

#     return result

# def to_trajectory(tokens):
#     out = []
#     i = 0
#     while i < len(tokens):
#         tok = tokens[i]
#         if len(tok) == 1 and tok.isalpha():
#             if i+1 < len(tokens) and re.match(r"^\d+degrees$", tokens[i+1]):
#                 angle = int(tokens[i+1].replace("degrees", ""))
#                 out.extend(angle_to_chars(tok, angle))
#                 i += 2
#             else:
#                 out.extend([tok.lower()] * 3)
#                 i += 1
#         else:
#             i += 1

#     filtered = []
#     for ch in out:
#         if len(filtered) >= 5 and all(c == ch for c in filtered[-5:]):
#             continue
#         filtered.append(ch)
#     return "".join(filtered)


# === New: Participant registration endpoint ===
@typingPage.route("/register", methods=["POST"])
def register_participant():
    data = request.get_json() or {}
    raw_id = data.get('id', '').strip()
    if not raw_id:
        return jsonify(error="Missing participant ID"), 400

    base = re.sub(r"[^\w-]", "_", raw_id.lower())

    try:
        with open(PARTICIPANTS_FILE, 'r') as f:
            participants = json.load(f)
    except (FileNotFoundError, JSONDecodeError):
        participants = []

    if base not in participants:
        new_id = base
    else:
        idx = 1
        while f"{base}_{idx}" in participants:
            idx += 1
        new_id = f"{base}_{idx}"

    participants.append(new_id)
    with open(PARTICIPANTS_FILE, 'w') as f:
        json.dump(participants, f, indent=2)

    return jsonify(id=new_id)

# === New: serve the study page ===
@typingPage.route("/study")
def study_page():
    return render_template('user-study.html')
    
@typingPage.route("/interface")
def interface():
    IMAGE_URL = url_for('.static', filename='js/keyboard_update.png')
    print(IMAGE_URL)
    return render_template('index.html', kBimage = IMAGE_URL)

@typingPage.route('/predict', methods=["POST"])
def legacy_predict_passthrough():
    # Back-compat for old JS that posts to /typing/predict
    try:
        payload = request.get_json(force=True)
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
        return jsonify(r.json())
    except Exception as e:
        return jsonify(error=str(e)), 500

@typingPage.route('/api/frontend/predict', methods=["POST"])
def frontend_predict():
    try:
        payload = request.get_json(force=True)
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=5)
        return jsonify(r.json())
    except Exception as e:
        return jsonify(error=str(e)), 500

@typingPage.route('/api/frontend/debug', methods=["POST"])
def frontend_debug():
    """Frontend predict"""
    try:
        string = request.json.get("string", "")
        print(string)
        return jsonify("")

    except Exception as e:
        return jsonify(error=str(e)), 500


app = Flask(__name__)
app.register_blueprint(typingPage, url_prefix='/typing')
CORS(app)

if __name__ == "__main__":
    app.run(port=1111, debug=True)
