import os
import math
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re

# === APP SETUP ===
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

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
            if dist < 0.15 and dist < min_dist:  # wider radius
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

def generate_predictions(input_str, beams=5, topn=3):
    inputs = tokenizer(input_str, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_length=8,
        num_beams=beams,
        num_return_sequences=topn,
        early_stopping=True
    )
    decoded = [
        tokenizer.decode(output, skip_special_tokens=True).replace("<pad>", "").strip().lower()
        for output in outputs
    ]
    return list(dict.fromkeys(decoded))[:topn]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        tokens = request.json.get("input", "").strip().split()
        if not tokens:
            return jsonify(error="Empty input."), 400
        
        if not tokens:
            return jsonify(error="Empty input."), 400

        try:
            letters = [t for t in tokens if len(t) == 1 and t.isalpha()]
            swipes = [t for t in tokens if re.match(r"^\d+degrees$", t)]
            if not swipes:
                typed_word = "".join(letters).lower()
                return jsonify(predictions=[typed_word], pattern=typed_word)
        except Exception as e:
            return jsonify(error=str(e)), 500

        trajectory = to_trajectory(tokens)
        print(f"Trajectory: {trajectory}")
        prompt = (
            "You are an intelligent QWERTY keyboard decoder. "
            "The input is the closest key sequence to the user-drawn gesture trajectory. "
            f"Please find the target word for this input: {trajectory}"
        )
        predictions = generate_predictions(prompt)
        return jsonify(predictions=predictions, pattern=trajectory)

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
