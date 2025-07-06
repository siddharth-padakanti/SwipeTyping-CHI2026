import math
import pandas as pd
import ast
import re
import numpy as np

# global variable
swipe_length_key = 3
key_coord_x = 0.2 # key distance in (-1, 1) coordinate
key_coord_y = 0.5 # key distance in (-1, 1) coordinate

tap_coords = {
    "Q": (-0.9, -0.72972973), "W": (-0.7, -0.72972973), "E": (-0.5, -0.72972973), "R": (-0.3, -0.72972973), "T": (-0.1, -0.72972973),
    "Y": (0.1, -0.72972973), "U": (0.3, -0.72972973), "I": (0.5, -0.72972973), "O": (0.7, -0.72972973), "P": (0.9, -0.72972973),
    "A": (-0.8, -0.18918919), "S": (-0.6, -0.18918919), "D": (-0.4, -0.18918919), "F": (-0.2, -0.18918919), "G": (0, -0.18918919),
    "H": (0.2, -0.18918919), "J": (0.4, -0.18918919), "K": (0.6, -0.18918919), "L": (0.8, -0.18918919),
    "Z": (-0.6, 0.35135135), "X": (-0.4, 0.35135135), "C": (-0.2, 0.35135135), "V": (0, 0.35135135), "B": (0.2, 0.35135135),
    "N": (0.4, 0.35135135), "M": (0.6, 0.35135135)
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
    angle = math.degrees(math.atan2(dy, dx))
    return int((angle + 360) % 360)

def get_swipe_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    d_distance = math.sqrt(dx*dx + dy*dy)
    # cap the calculated swipe key
    x_res = x1 + dx * swipe_length_key / d_distance
    y_res = y1 + dy * swipe_length_key / d_distance
    return x_res, y_res

def get_trajectory_chars(entry_x, entry_y):
    chars_res = []

    if len(entry_x) == 1:
        # only one point, return same 25 characters
        for t in range(25):
            chars_res.append(find_closest_key(entry_x[0], entry_y[0]))
    else:
        # get full distance
        path_dis = 0
        original_axis = []
        original_axis.append(0)
        for i in range(len(entry_x) - 1):
            dx = entry_x[i + 1] - entry_x[i]
            dy = entry_y[i + 1] - entry_y[i]
            path_dis += math.sqrt(dx*dx + dy*dy)
            original_axis.append(path_dis)

        interp_x = np.interp(np.linspace(0, path_dis, num=25), xp = original_axis, fp = entry_x)
        interp_y = np.interp(np.linspace(0, path_dis, num=25), xp = original_axis, fp = entry_y)

        #print(original_axis)
        #print(np.linspace(0, path_dis, num=25))
        #print(interp_x)
        #print(interp_y)
        for t in range(len(interp_x)):
            chars_res.append(find_closest_key(interp_x[t], interp_y[t]))

    return chars_res
    

def process_csv(filepath):

    print("-------------------------------")

    df = pd.read_csv(filepath)
    results = []

    for _, row in df.iterrows():
        x_input = ast.literal_eval(row['x'])
        y_input = ast.literal_eval(row['y'])
        entry_result_x = []
        entry_result_y = []


        for i in range(len(x_input)):
            x_segment = x_input[i]
            y_segment = y_input[i]

            if len(x_segment) == 1:
                entry_result_x.append(x_segment[0])
                entry_result_y.append(y_segment[0])
            elif len(x_segment) >= 2:
                x1, y1 = x_segment[0], y_segment[0]
                x2, y2 = x_segment[-1], y_segment[-1]

                x1_key = x1 / key_coord_x
                y1_key = y1 / key_coord_y
                x2_key = x2 / key_coord_x
                y2_key = y2 / key_coord_y

                print(f"Tap Coordinates: ({x1},{y1}) and ({x2}, {y2}); Tap key: ({x1_key},{y1_key}) and ({x2_key}, {y2_key})")
                angle = get_angle(x1_key, y1_key, x2_key, y2_key)
                print(f"{angle}degrees")

                x2_swipe_key, y2_swipe_key = get_swipe_key(x1_key, y1_key, x2_key, y2_key)
                x2_swipe = x2_swipe_key * key_coord_x
                y2_swipe = y2_swipe_key * key_coord_y
                print(f"Swipe Coordinates: ({x1},{y1}) and ({x2_swipe}, {y2_swipe})")
                entry_result_x.append(x1)
                entry_result_x.append(x2_swipe)
                entry_result_y.append(y1)
                entry_result_y.append(y2_swipe)

        print("------------Path---------------")

        print(f"{row['word']} ({row['sub_gestures']})")

        print(f"{entry_result_x}")
        print(f"{entry_result_y}")

        char_res = get_trajectory_chars(entry_result_x, entry_result_y)

        print("Trajectory:", char_res)

        print("-------------------------------")

# Example usage
process_csv("test2.csv")
