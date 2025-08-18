import math
import pandas as pd
import ast
import numpy as np
import os

length_option = [2, 3, 4]
fold_option = 10

key_coord = 80
tap_coords = {
    "Q": (40, 40), "W": (120, 40), "E": (200, 40), "R": (280, 40), "T": (360, 40),
    "Y": (440, 40), "U": (520, 40), "I": (600, 40), "O": (680, 40), "P": (760, 40), "backspace": (840, 40),
    "A": (60, 120), "S": (140, 120), "D": (220, 120), "F": (300, 120), "G": (380, 120),
    "H": (460, 120), "J": (540, 120), "K": (620, 120), "L": (700, 120), "enter": (810, 120),
    "Z": (100, 200), "X": (180, 200), "C": (260, 200), "V": (340, 200), "B": (420, 200),
    "N": (500, 200), "M": (580, 200), ",": (660, 200), ".": (740, 200), " ": (440, 280)
}

# output
len_idx, fold_idx = [], []
char_idx = []
accuracy_top1 = []
accuracy_top3 = []

def process():
    for length in length_option:
        for fold in range(fold_option):
            print(str(length) + ", " + str(fold))
            # read csv
            filepath = "./swipe_length_" + str(length) + "/fold_" + str(fold) + "/test_result.csv"

            df = pd.read_csv(filepath)
            # print(df)

            correct_top1 = 0
            correct_top3 = 0
            total = 0

            for idx, row in df.iterrows():
                if row["correct"] == True:
                    correct_top1 += 1
                if row["corrects_top3"] == True:
                    correct_top3 += 1
                total += 1

            accu_top1 = correct_top1 / total
            accu_top3 = correct_top3 / total

            len_idx.append(length)
            fold_idx.append(fold)
            accuracy_top1.append(accu_top1)
            accuracy_top3.append(accu_top3)

    pd.DataFrame({"length": len_idx, "fold": fold_idx, "accuracy_top1": accuracy_top1, "accuracy_top3": accuracy_top3}).to_csv("fold_accuracy.csv", index=False)

def process_char_count():
    for length in length_option:
        for fold in range(fold_option):
            
            print(str(length) + ", " + str(fold))
            # read csv
            filepath = "./swipe_length_" + str(length) + "/fold_" + str(fold) + "/test_result.csv"

            df = pd.read_csv(filepath)
            # print(df)

            correct_top1 = []
            correct_top3 = []
            total = []

            for char_count in range(20):
                correct_top1.append(0)
                correct_top3.append(0)
                total.append(0)

            for idx, row in df.iterrows():

                if row["correct"] == True:
                    correct_top1[int(row["count"]) - 1] += 1
                if row["corrects_top3"] == True:
                    correct_top3[int(row["count"]) - 1] += 1
                total[int(row["count"]) - 1] += 1

            for char_count in range(20):
                if total[char_count] != 0:
                    accu_top1 = correct_top1[char_count] / total[char_count]
                    accu_top3 = correct_top3[char_count] / total[char_count]

                    len_idx.append(length)
                    fold_idx.append(fold)
                    char_idx.append(char_count + 1)
                    accuracy_top1.append(accu_top1)
                    accuracy_top3.append(accu_top3)

    pd.DataFrame({"length": len_idx, "fold": fold_idx, "char_num": char_idx, "accuracy_top1": accuracy_top1, "accuracy_top3": accuracy_top3}).to_csv("fold_accuracy_char_count.csv", index=False)


def calculate_character_gap():
    df = pd.read_csv("word.csv", keep_default_na=False)

    key_dist = []
    for idx, row in df.iterrows():
        print(row)
        for i in range(row["count"] - 1):
            char = row["target"].upper()[i]

            (x, y) = tap_coords[char]

            char2 = row["target"].upper()[i + 1]

            (x2, y2) = tap_coords[char2]

            #print(str(x) + ", " + str(y))


            dx = x2 - x
            dy = y2 - y

            swipe_len_dist = math.sqrt(dx*dx + dy*dy) / key_coord
            #print(swipe_len_dist)

            key_dist.append(swipe_len_dist)

    print("average key distance: ")
    print(np.mean(key_dist))



if __name__ == "__main__":
    # process()
    process_char_count()
    # calculate_character_gap()