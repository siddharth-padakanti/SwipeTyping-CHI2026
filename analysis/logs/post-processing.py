# parse the log file into main results
import math
import pandas as pd
import ast
import numpy as np
import os
from datetime import datetime

participant_id = [2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 23]

tap_coords = {
    "Q": (40, 40), "W": (120, 40), "E": (200, 40), "R": (280, 40), "T": (360, 40),
    "Y": (440, 40), "U": (520, 40), "I": (600, 40), "O": (680, 40), "P": (760, 40), "backspace": (840, 40),
    "A": (60, 120), "S": (140, 120), "D": (220, 120), "F": (300, 120), "G": (380, 120),
    "H": (460, 120), "J": (540, 120), "K": (620, 120), "L": (700, 120), "enter": (810, 120),
    "Z": (100, 200), "X": (180, 200), "C": (260, 200), "V": (340, 200), "B": (420, 200),
    "N": (500, 200), "M": (580, 200), ",": (660, 200), ".": (740, 200), " ": (440, 280)
}
key_coord = 80

def isInTimeRange(time, start, end):
    FMT = "%H:%M:%S.%f"
    start_time = (datetime.strptime(time, FMT) - datetime.strptime(start, FMT)).total_seconds()
    end_time = (datetime.strptime(time, FMT) - datetime.strptime(end, FMT)).total_seconds()
    return (datetime.strptime(time, FMT) and end_time <= 0)

def process_trial(pid, tid, gesture, word, trial):
    print(tid)
    # print(gesture)
    # print(word)
    # print(trial)

    # Word per minutes: start from the first gesture in the trial, to the trial end.
    trial_start_time = gesture.iloc[0]['Time']
    trial_end_time = trial.iloc[-1]['Time']

    # print(trial_start_time)
    # print(trial_end_time)
    trial_time = (trial_end_time - trial_start_time).total_seconds()
    # print(trial_time)

    trial_char_count = len(trial.iloc[0]['Target'])
    # print(trial_char_count)
    wpm = (trial_char_count - 1) / trial_time * 60 / 5
    # print(wpm)

    # Uncorrected error rate: 
    # Corrected error rate
    target_string = trial.iloc[-1]['Target'].split()
    input_string = trial.iloc[-1]['Current Sentence'].split()
    wordCount = len(target_string)
    
    uncorrected_error = 0
    corrected_word = 0
    for i in range(wordCount):
        if(target_string[i] != input_string[i]):
            uncorrected_error += 1
        else:
            corrected_word += 1

    corrected_error = 0
    backspace_df = trial[(trial['Action'] == 'DeletePartial') | (trial['Action'] == 'DeleteFull')]
    for idx, row in backspace_df.iterrows():
        before_string = row['Current Sentence'].split()
        after_string = row['Updated Sentence'].split()
        if len(before_string) != len(after_string):
            corrected_error += 1
    
    # print(corrected_word)
    # print(uncorrected_error)
    # print(corrected_error)
    
    uwer = uncorrected_error / (corrected_word + uncorrected_error + corrected_error)
    cwer = corrected_error / (corrected_word + uncorrected_error + corrected_error)
    # print([uwer, cwer])

    trial_result = [{
        "pid": pid,
        "tid": tid, 
        "wpm": wpm,
        "corrected_word": corrected_word,
        "uncorrected_error": uncorrected_error,
        "corrected_error": corrected_error,
        "uwer": uwer,
        "cwer": cwer,
    }]

    # loop for every word, find out top1/top3/error, mapping word usage, word length    
    # top1 accuracy through word length
    # top3 accuracy through word length
    # swipe typing/ tap only ratio per user
    # swipe typing/ tap only ratio per word length

    insert_word = []
    # insert_df = trial[(trial['Action'] == 'Insert_top1') | (trial['Action'] == 'Insert_top3')]
    insert_start_time = trial.iloc[0]['Time']
    for idx, row in trial.iterrows():
        target_string = row['Target'].split()
        if ((row['Action'] == 'Insert_top1') or (row['Action'] == 'Insert_top3')):
            before_string = row['Current Sentence'].split()
            after_string = row['Updated Sentence'].split()
            # print(target_string)
            # print(before_string)
            # print(after_string)
            if len(after_string) > len(insert_word):
                category = "incorrect"
                if target_string[len(after_string) - 1] == after_string[len(after_string) - 1]:
                    if row['Action'] == 'Insert_top1':
                        category = "top1"
                    elif row['Action'] == 'Insert_top3':
                        category = "top3"
                
                insert_word.append({
                    "pid": pid,
                    "target": target_string[len(after_string) - 1],
                    "word": after_string[len(after_string) - 1],
                    "count": len(target_string[len(after_string) - 1]),
                    "insert_start_time": insert_start_time,
                    "insert_end_time": row['Time'],
                    "category": category
                    })
            else:
                category = "incorrect"
                if target_string[len(after_string) - 1] == after_string[len(after_string) - 1]:
                    if row['Action'] == 'Insert_top1':
                        category = "top1"
                    elif row['Action'] == 'Insert_top3':
                        category = "top3"
                insert_word[len(after_string) - 1] = {
                    "pid": pid,
                    "target": target_string[len(after_string) - 1],
                    "word": after_string[len(after_string) - 1],
                    "count": len(target_string[len(after_string) - 1]),
                    "insert_start_time": insert_start_time,
                    "insert_end_time": row['Time'],
                    "category": category
                    }   
                
        # extra work for the pilot logs to parse the auto complete word
        if ((row['Action'] == 'End') and (len(target_string) != len(insert_word))):
            after_string = row['Updated Sentence'].split()
            insert_word.append({
                "pid": pid,
                "target": target_string[len(after_string) - 1],
                "word": after_string[len(after_string) - 1],
                "count": len(target_string[len(after_string) - 1]),
                "insert_start_time": insert_start_time,
                "insert_end_time": row['Time'],
                "category": "top1"
            })

        insert_start_time = row['Time']
    # print(insert_word)

    for idx in range(len(insert_word)):
        insert = insert_word[idx]
        # print(insert)

        gesture_insertdf = gesture[(gesture['Time'] >= insert["insert_start_time"]) & (gesture['Time'] <= insert["insert_end_time"])]
        word_insertdf = word[(word['Time'] >= insert["insert_start_time"]) & (word['Time'] <= insert["insert_end_time"])]

        # print(gesture_insertdf)
        # print(word_insertdf)

        # within swipe typing: swipe/ tap ratio per word
        # within swipe typing: swipe distance
        # within swipe typing: swipe key sets
        swipe_count = 0
        target = insert["target"]
        target_char_count = 0
        swipe_dist = []
        swipe_key_set = []
        tap_count = 0
        for gidx, grow in gesture_insertdf.iterrows():
            if grow['Type'] == "swipe":
                if target_char_count + 1 < len(target):
                    swipe_count += 1
                    swipe_key_set.append([target[target_char_count], target[target_char_count+1]])
                    tarX1 = tap_coords[target[target_char_count].upper()][0]
                    tarY1 = tap_coords[target[target_char_count].upper()][1]
                    tarX2 = tap_coords[target[target_char_count+1].upper()][0]
                    tarY2 = tap_coords[target[target_char_count+1].upper()][1]
                    dtx = tarX2 - tarX1
                    dty = tarY2 - tarY1
                    distarget = math.sqrt(dtx*dtx + dty*dty) / key_coord
                    dx = float(grow['EndX']) - float(grow['StartX'])
                    dy = float(grow['EndY']) - float(grow['StartY'])
                    dis = math.sqrt(dx*dx + dy*dy) / key_coord
                    swipe_dist.append([dis, distarget])
                
                target_char_count += 2
            elif grow['Type'] == "tap":
                if target_char_count  < len(target):
                    tap_count += 1
                target_char_count += 1

        # print(swipe_count)
        # print(tap_count)
        
        swipe_ratio = swipe_count / (swipe_count + tap_count)
        tap_ratio = tap_count / (swipe_count + tap_count)

        insert["swipe_ratio"] = swipe_ratio
        insert["tap_ratio"] = tap_ratio
        insert["swipe_key_count"] = swipe_count * 2
        insert["tap_key_count"] = tap_count
        insert["swipe_dist"] = swipe_dist
        insert["swipe_key_set"] = swipe_key_set

        # print(swipe_ratio)
        # print(tap_ratio)
        # print(swipe_dist)
        # print(swipe_key_set)


        
        # within swipe typing: swipe finger

    # print(insert_word)

    return trial_result, insert_word










if __name__ == '__main__':
    total_result = []
    total_usage = []
    total_word = []
    for pid in participant_id:
        print(pid)
        gesture_df = pd.read_csv(str(pid) + '/gesture_log.csv', keep_default_na=False)
        trial_df = pd.read_csv(str(pid) + '/trial_log.csv', keep_default_na=False)
        word_df = pd.read_csv(str(pid) + '/word_log.csv', keep_default_na=False)

        gesture_df['Time'] = pd.to_datetime(gesture_df['Time'], format="%H:%M:%S.%f")
        trial_df['Time'] = pd.to_datetime(trial_df['Time'], format="%H:%M:%S.%f")
        word_df['Time'] = pd.to_datetime(word_df['Time'], format="%H:%M:%S.%f")

        participant_word = []

        for trial_id in range(30):
            trial_result, words = process_trial(pid, trial_id, gesture_df[gesture_df['Trial Num'] == trial_id], 
                          word_df[word_df['Trial Num'] == trial_id], trial_df[trial_df['Trial Num'] == trial_id])
            
            total_result = total_result + trial_result
            participant_word = participant_word + words

        # calculate results per participants
        # print(participant_word)

        swipeTyping_count = 0
        tapTyping_count = 0
        for idx in range(len(participant_word)):
            word = participant_word[idx]
            if word['tap_ratio'] == 1:
                tapTyping_count += 1
            else:
                swipeTyping_count += 1

        total_usage.append({
            "pid": pid,
            "swipeTyping_usage": swipeTyping_count / (swipeTyping_count + tapTyping_count)
        })

        total_word = total_word + participant_word
    

    # save out results csv
    resultdf = pd.DataFrame(total_result)
    usagedf = pd.DataFrame(total_usage)
    worddf = pd.DataFrame(total_word)

    resultdf.to_csv('result.csv', index=False)  
    usagedf.to_csv('usage.csv', index=False)  
    worddf.to_csv('word.csv', index=False)  


