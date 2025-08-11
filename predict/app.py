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

# === APP SETUP ===
app = Flask(__name__)
CORS(app, origins=["http://localhost:1111", "http://127.0.0.1:1111"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model_path = "./model/"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

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

def angle_to_chars(char, angle):
    char = char.upper()
    if char not in keyboard_layout:
        return []

    cx, cy = keyboard_layout[char]
    angle = (angle + 360) % 360
    radians = math.radians(angle)
    dx = math.cos(radians)
    dy = math.sin(radians)

    result = [char.lower()] * 3
    visited = set(result)

    for step in range(1, 15):  # allow longer reach
        fx = cx + dx * step * 0.2
        fy = cy + dy * step * 0.2

        closest = None
        min_dist = float("inf")

        for k, (kx, ky) in keyboard_layout.items():
            if k.lower() in visited:
                continue
            dist = (fx - kx)**2 + (fy - ky)**2
            if dist < 0.2 and dist < min_dist:  # wider radius
                closest = k.lower()
                min_dist = dist

        if closest:
            visited.add(closest)
            multiplier = 5 if closest in "aeiou" else 2
            result.extend([closest] * multiplier)
            if len(visited) >= 5:
                break

    return result

def to_trajectory(tokens):
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if len(tok) == 1 and tok.isalpha():
            if i+1 < len(tokens) and re.match(r"^\d+degrees$", tokens[i+1]):
                angle = int(tokens[i+1].replace("degrees", ""))
                out.extend(angle_to_chars(tok, angle))
                i += 2
            else:
                out.extend([tok.lower()] * 3)
                i += 1
        else:
            i += 1

    filtered = []
    for ch in out:
        if len(filtered) >= 5 and all(c == ch for c in filtered[-5:]):
            continue
        filtered.append(ch)
    return "".join(filtered)


def generate_n_best_words(text, count, num_beams=11, num_return_sequences=10, max_new=None):
    count = int(count)
    max_new = max_new or max(count, 1)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs['input_ids'],
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new,
        min_new_tokens=1,         
        early_stopping=True,
        # length_penalty=0.0,     
    )

    words = [
        tokenizer.decode(o, skip_special_tokens=True)
        .replace("</s>", "").replace("<pad>", "").replace("<unk>", "").replace(" ", "")
        for o in outputs
    ]
    filtered = [w for w in words if len(w) == count]
    return list(dict.fromkeys(filtered))[:3]

# === New: Participant registration endpoint ===
@app.route("/register", methods=["POST"])
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

        
@app.route('/predict', methods=["POST"])
def predict():
    try:
        tokens = request.json.get("input", "")
        count = request.json.get("count", "")
        if not tokens:
            return jsonify(error="Empty input."), 400
        
        print(datetime.now())
        #print(tokens)
        # try:
        #     letters = [t for t in tokens if len(t) == 1 and t.isalpha()]
        #     swipes = [t for t in tokens if re.match(r"^\d+degrees$", t)]
        #     if not swipes:
        #         typed_word = "".join(letters).lower()
        #         return jsonify(predictions=[typed_word], pattern=typed_word)
        # except Exception as e:
        #     return jsonify(error=str(e)), 500

        # trajectory = to_trajectory(tokens)

        #Handle Taps-only
        # onlyTaps = request.json.get("tapsOnly", "")
        
        # if onlyTaps:
        #     print("no swipes")
        #     typed_word = request.json.get("word", "")
        #     return jsonify(predictions=[typed_word], pattern=typed_word)
        # else:
        # Else: Handle tap+swipe input
        prompt = (
            "You are an intelligent QWERTY keyboard decoder. "
            "The input is the closest key sequence to the user-drawn gesture trajectory. "
            f"Please only find the {count}-LETTER target word for this input: {tokens}"
        )
        sequence = (request.json.get("input", "") or "").lower()
        count = int(request.json.get("count", 0))
        if not sequence or count <= 0:
            return jsonify(error="Empty input or invalid count."), 400

        predictions = generate_n_best_words(sequence, count)
        print(datetime.now())
        return jsonify(predictions=predictions, pattern=sequence)

    except Exception as e:
        return jsonify(error=str(e)), 500
    

@app.route('/debug', methods=["POST"])
def debug():
    try:
        string = request.json.get("string", "")
        print(string)
        return jsonify("")

    except Exception as e:
        return jsonify(error=str(e)), 500






if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True)
