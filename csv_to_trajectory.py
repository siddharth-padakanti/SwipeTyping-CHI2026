import math
import pandas as pd
import ast
import re


tap_coords = {
    "Q": (-0.9, -0.72972973), "W": (-0.7, -0.72972973), "E": (-0.5, -0.72972973), "R": (-0.3, -0.72972973), "T": (-0.1, -0.72972973),
    "Y": (0.1, -0.72972973), "U": (0.3, -0.72972973), "I": (0.5, -0.72972973), "O": (0.7, -0.72972973), "P": (0.9, -0.72972973),
    "A": (-0.8, -0.18918919), "S": (-0.6, -0.18918919), "D": (-0.4, -0.18918919), "F": (-0.2, -0.18918919), "G": (0, -0.18918919),
    "H": (0.2, -0.18918919), "J": (0.4, -0.18918919), "K": (0.6, -0.18918919), "L": (0.8, -0.18918919),
    "Z": (-0.6, 0.35135135), "X": (-0.4, 0.35135135), "C": (-0.2, 0.35135135), "V": (0, 0.35135135), "B": (0.2, 0.35135135),
    "N": (0.4, 0.35135135), "M": (0.6, 0.35135135)
}

swipe_coords = {
    "Q": (-4.5, 0.75), "W": (-3.5, 0.75), "E": (-2.5, 0.75), "R": (-1.5, 0.75), "T": (-0.5, 0.75),
    "Y": (0.5, 0.75), "U": (1.0, 0.75), "I": (1.5, 0.75), "O": (2.0, 0.75), "P": (2.5, 0.75),
    "A": (-4, 0), "S": (-3, 0), "D": (-2, 0), "F": (-1, 0), "G": (0, 0),
    "H": (1, 0), "J": (2, 0), "K": (3, 0), "L": (4, 0),
    "Z": (-3.5, -0.75), "X": (-2.5, -0.75), "C": (-1.5, -0.75), "V": (-0.5, -0.75), "B": (0.5, -0.75),
    "N": (1.5, -0.75), "M": (2.5, -0.75)
}

def find_closest_key(x, y):
    min_dist = float("inf")
    closest = None
    for key, (kx, ky) in tap_coords.items():
        dist = (x - kx) ** 2 + (y - ky) ** 2
        if dist < min_dist:
            min_dist = dist
            closest = key
    return closest

def get_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy))
    return int((angle + 360) % 360)

# Manual QWERTY layout using physical positions (unit spacing)
us_keys = [
    ("Q", 0, 0), ("W", 1, 0), ("E", 2, 0), ("R", 3, 0), ("T", 4, 0),
    ("Y", 5, 0), ("U", 6, 0), ("I", 7, 0), ("O", 8, 0), ("P", 9, 0),
    ("A", 0.5, 1), ("S", 1.5, 1), ("D", 2.5, 1), ("F", 3.5, 1), ("G", 4.5, 1),
    ("H", 5.5, 1), ("J", 6.5, 1), ("K", 7.5, 1), ("L", 8.5, 1),
    ("Z", 1, 2), ("X", 2, 2), ("C", 3, 2), ("V", 4, 2), ("B", 5, 2),
    ("N", 6, 2), ("M", 7, 2)
]

key_coords = {char: (x, y) for char, x, y in us_keys}

# Find valid target letters in a given angle direction

def angle_to_direction_chars(char, angle):
    char = char.upper()
    if char not in key_coords:
        return []

    base_x, base_y = key_coords[char]
    angle = (angle + 360) % 360
    radians = math.radians(angle)
    dx = math.sin(radians)  # horizontal movement (right)
    dy = -math.cos(radians)  # vertical movement (up)

    result = [char.lower()] * 3
    visited = set([char.lower()])

    steps_taken = 0
    for step in range(1, 10):
        fx = base_x + dx * step
        fy = base_y + dy * step

        closest_key = None
        closest_dist = float("inf")
        for k, (kx, ky) in key_coords.items():
            dist = (fx - kx)**2 + (fy - ky)**2
            if dist < 0.5 and k.lower() not in visited and dist < closest_dist:
                closest_key = k.lower()
                closest_dist = dist

        if closest_key:
            visited.add(closest_key)
            multiplier = 5 if closest_key in "aeiou" else 2
            result.extend([closest_key] * multiplier)
            steps_taken += 1
            if steps_taken == 3:
                break

    return result

# Converts tokens into a trajectory string
def to_trajectory_string(tokens):
    gesture_chars = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if len(token) == 1 and token.isalpha():
            base_char = token.upper()
            if i + 1 < len(tokens) and re.match(r"^\d+degrees$", tokens[i + 1]):
                angle = int(tokens[i + 1].replace("degrees", ""))
                chars = angle_to_direction_chars(base_char, angle)
                gesture_chars.extend(chars)
                i += 2
            else:
                gesture_chars.extend([base_char.lower()] * 3)
                i += 1
        else:
            i += 1

    # Limit excessive repetition: max 5 of same char in a row
    filtered = []
    for char in gesture_chars:
        if len(filtered) >= 5 and all(c == char for c in filtered[-5:]):
            continue  # skip adding if last 5 are same as current
        filtered.append(char)

    return "".join(filtered)

def process_csv(filepath):

    print("-------------------------------")

    df = pd.read_csv(filepath)
    results = []

    for _, row in df.iterrows():
        x_input = ast.literal_eval(row['x'])
        y_input = ast.literal_eval(row['y'])
        entry_result = []

        for i in range(len(x_input)):
            x_segment = x_input[i]
            y_segment = y_input[i]

            if len(x_segment) == 1:
                key = find_closest_key(x_segment[0], y_segment[0])
                entry_result.append(key.upper())
            elif len(x_segment) >= 2:
                x1, y1 = x_segment[0], y_segment[0]
                x2, y2 = x_segment[-1], y_segment[-1]
                first_key = find_closest_key(x1, y1)
                last_key = find_closest_key(x2, y2)
                print(f"Tap Coordinates: ({x1},{y1}) and ({x2}, {y2})")
                print(f"Swipe Coordinates: ({swipe_coords[first_key][0]},{swipe_coords[first_key][1]}) and ({swipe_coords[last_key][0]}, {swipe_coords[last_key][1]})")
                angle = get_angle(swipe_coords[first_key][0], swipe_coords[first_key][1], swipe_coords[last_key][0], swipe_coords[last_key][1])
                entry_result.append(f"{first_key.upper()} {angle}degrees")

        print(f"{row['word']} ({row['sub_gestures']}): {' '.join(entry_result)}")

        print(f"{entry_result}")

        user_input = ' '.join(entry_result)
        tokens = user_input.strip().split()
    
        print("Trajectory:", to_trajectory_string(tokens))
        print("-------------------------------")

# Example usage
process_csv("test2.csv")
