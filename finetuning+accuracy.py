import math
import pandas as pd
import ast
import re
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import (Dataset, load_dataset)
from transformers import TrainerCallback
import os

# global variable
swipe_length_key = 3
key_coord_x = 0.2 # key distance in (-1, 1) coordinate
key_coord_y = 0.5 # key distance in (-1, 1) coordinate

# Load local model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./"  # Folder with model files

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

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

def generate_n_best_words(text, num_beams=5, num_return_sequences=4):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_length=8,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True
    )
    words = [
        tokenizer.decode(output, skip_special_tokens=True)
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("<unk>", "")
        .replace(" ", "")
        for output in outputs
    ]
    return list(dict.fromkeys(words))[:3]

def create_training_pairs(csv_file, output_file="finetune_data.csv"):
    print(f"Extraction from {csv_file} initiated")
    df = pd.read_csv(csv_file, keep_default_na=False)
    inputs, targets = [], []

    for _, row in df.iterrows():
        x_input = ast.literal_eval(row['x'])
        y_input = ast.literal_eval(row['y'])
        entry_result_x, entry_result_y = [], []

        for i in range(len(x_input)):
            x_seg = x_input[i]
            y_seg = y_input[i]
            if len(x_seg) == 1:
                entry_result_x.append(x_seg[0])
                entry_result_y.append(y_seg[0])
            elif len(x_seg) >= 2:
                x1, y1 = x_seg[0], y_seg[0]
                x2, y2 = x_seg[-1], y_seg[-1]
                x1_key, y1_key = x1 / key_coord_x, y1 / key_coord_y
                x2_key, y2_key = x2 / key_coord_x, y2 / key_coord_y
                x2_swipe, y2_swipe = get_swipe_key(x1_key, y1_key, x2_key, y2_key)
                entry_result_x += [x1, x2_swipe * key_coord_x]
                entry_result_y += [y1, y2_swipe * key_coord_y]

        trajectory_string = "".join(get_trajectory_chars(entry_result_x, entry_result_y))
        inputs.append(trajectory_string)
        targets.append(str(row["word"]))
        print(f"Extraction complete for the word: '{row["word"]}'")

    pd.DataFrame({"input": inputs, "target": targets}).to_csv(output_file, index=False)
    print(f"Saved fine-tune data to {output_file}") 

def preprocess(example):
    input_raw  = example.get("input", "").strip()
    target_raw = example.get("target", "").strip()
    if not input_raw or not target_raw:
        return None

    input_text  = "Find word from trajectory: " + input_raw
    target_text = target_raw

    model_inputs = tokenizer(
        text=input_text,
        max_length=32,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        text_target=target_text,
        max_length=8,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class CleanSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Strip out any extra keys (like num_items_in_batch)  
        inputs = inputs.copy()
        inputs.pop("num_items_in_batch", None)
        return super().compute_loss(model, inputs, return_outputs)
    
def cross_validate_finetune(
    data_path: str,
    output_base_dir: str,
    n_splits: int = 10,
    test_size: float = 0.1,
):
    # 1) Load CSV via pandas, preserving "null"/"none" as strings
    df = pd.read_csv(
        data_path,
        keep_default_na=False,  # don't convert "null"/"none" to NaN
        na_values=[]            # no extra NA tokens
    )

    # 2) Convert to a Hugging Face Dataset
    raw_ds = Dataset.from_pandas(df.reset_index(drop=True))

    # 3) 10× random 90/10 splits
    for fold in range(n_splits):

        fold_dir = os.path.join(output_base_dir, f"fold_{fold+1}")
        # ── SKIP IF THIS FOLD IS ALREADY DONE ──────────────
        marker = os.path.join(fold_dir, "DONE")
        if os.path.exists(marker):
            print(f"✅ Fold {fold+1} already complete, skipping.")
            continue
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        split    = raw_ds.shuffle(seed=fold).train_test_split(test_size=test_size)
        train_ds = split["train"]
        eval_ds  = split["test"]

        # tokenize
        tok_train = train_ds.map(
            preprocess,
            remove_columns=train_ds.column_names,
            num_proc=1,
        )
        tok_eval = eval_ds.map(
            preprocess,
            remove_columns=eval_ds.column_names,
            num_proc=1,
        )

        # 1) pick a batch_size variable
        batch_size = 16

        # 2) compute steps_per_epoch before touching training_args
        steps_per_epoch = len(tok_train) // batch_size

        # 3) now build your args
        training_args = Seq2SeqTrainingArguments(
            output_dir=fold_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=5,
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_dir=os.path.join(fold_dir, "logs"),

            # old‐style eval/save flags
            do_eval=True,
            eval_steps=steps_per_epoch,
            logging_steps=steps_per_epoch,
            save_steps=steps_per_epoch,

            save_total_limit=2,
            predict_with_generate=True,
            bf16 = True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tok_train,
            eval_dataset=tok_eval,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"Fold {fold+1} metrics:", metrics)

        model.save_pretrained(fold_dir)
        tokenizer.save_pretrained(fold_dir)

        # mark done
        open(marker, "w").close()


    # 4) Save the final model snapshot
    final_dir = os.path.join(output_base_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("✅ Cross-validation done. Final model in:", final_dir)

def evaluate_cross_validation(
    data_path: str,
    model_base_dir: str,
    n_splits: int = 10,
    test_size: float = 0.1,
):
    # 1) Load your CSV
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for fold in range(n_splits):
        # 2) Recreate the 10% eval split
        df_eval = df.sample(frac=test_size, random_state=fold).reset_index(drop=True)

        # 3) Load fold-specific model & tokenizer
        fold_dir = os.path.join(model_base_dir, f"fold_{fold+1}")
        tokenizer = AutoTokenizer.from_pretrained(fold_dir)
        fold_model = T5ForConditionalGeneration.from_pretrained(fold_dir).to(device)

        # 4) Inference loop
        correct = 0
        for _, row in df_eval.iterrows():
            prompt = "Find word from trajectory: " + row["input"]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
            outputs = fold_model.generate(
                inputs["input_ids"],
                max_length=8,
                num_beams=5,
                early_stopping=True
            )
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if pred == row["target"]:
                correct += 1

        accuracy = correct / len(df_eval) if len(df_eval) else 0.0
        results.append({"fold": fold+1, "accuracy": accuracy})

    # 5) Print results
    df_acc = pd.DataFrame(results)
    print("\nCross-Validation Accuracies:")
    print(df_acc.to_string(index=False))
    return df_acc

class PercentageCheckpointCallback(TrainerCallback):
    def __init__(self, trainer_ref, total_examples, save_every_percent=5, save_dir="./partial_checkpoints"):
        self.trainer_ref = trainer_ref
        self.total_examples = total_examples
        self.save_every_percent = save_every_percent
        self.examples_seen = 0
        self.next_checkpoint = (save_every_percent / 100.0) * total_examples
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        self.examples_seen += args.per_device_train_batch_size * args._n_gpu if torch.cuda.is_available() else args.per_device_train_batch_size

        if self.examples_seen >= self.next_checkpoint:
            pct = int((self.examples_seen / self.total_examples) * 100)
            checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_{pct}pct")

            print(f"\n>>> Saving {pct}% checkpoint to {checkpoint_dir}")
            self.trainer_ref.save_model(checkpoint_dir)
            self.trainer_ref.tokenizer.save_pretrained(checkpoint_dir)

            self.next_checkpoint += (self.save_every_percent / 100.0) * self.total_examples

def finetune_model(data_path, output_dir="./finetuned-flan-model"):
    dataset = load_dataset("csv", data_files=data_path)["train"]
    split_dataset = dataset.train_test_split(test_size=0.2)

    def preprocess(example):
        try:
            input_raw = example.get("input", "")
            target_raw = example.get("target", "")

            input_text = "Find word from trajectory: " + str(input_raw if input_raw else "none").strip()
            target_text = str(target_raw if target_raw else "none").strip()

            if not input_text or not target_text:
                print(f"Skipping empty or invalid: {example}")
                return None
            
            tokenized_input = tokenizer(
                text=input_text,
                max_length=32,
                truncation=True,
                padding="max_length"
            )

            tokenized_target = tokenizer(
                text_target=target_text,
                max_length=8,
                truncation=True,
                padding="max_length"
            )

            tokenized_input["labels"] = tokenized_target["input_ids"]
            return tokenized_input

        except Exception as e:
            print(f"Error processing example: {example} — {e}")
            return None  



    tokenized_train = split_dataset["train"].map(
        preprocess,
        num_proc=1,
        remove_columns=split_dataset["train"].column_names
    )

    tokenized_eval = split_dataset["test"].map(
        preprocess,
        remove_columns=split_dataset["test"].column_names
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        predict_with_generate=True,
        save_total_limit=2,
        bf16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[]  
    )

    total_training_examples = len(tokenized_train)
    checkpoint_callback = PercentageCheckpointCallback(
        trainer_ref=trainer,
        total_examples=total_training_examples,
        save_every_percent=5,  
        save_dir=os.path.join(output_dir, "partial_checkpoints")
    )
    trainer.add_callback(checkpoint_callback)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")   

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

        input_text = (
            "You are an intelligent QWERTY keyboard decoder. "
            "The input is the closest key sequence to the user-drawn gesture trajectory. "
            "Please find the target word for this input: " + "".join(char_res)
        )

        print(f"Predicted words: {generate_n_best_words(input_text)}")

        print("-------------------------------")

# Example usage
# process_csv("test2.csv")

# Extracting inputs and targets from csv files for training
# create_training_pairs("csv_files/trajectories_50.csv")


# df = pd.read_csv("finetune_data.csv")
# df.dropna(subset=["input", "target"], inplace=True)
# df = df[df["input"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
# df = df[df["target"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
# df.to_csv("finetune_data_clean.csv", index=False)

# Training and exporting data 
# finetune_model("finetune_data.csv")

if __name__ == "__main__":
    # 6a) regenerate your CSV as before:
    # create_training_pairs("csv_files/trajectories_50.csv")
    # … any cleaning steps you do …
    # 6b) then launch our 10-fold loop:
    # cross_validate_finetune(
    #     data_path="finetune_data.csv",
    #     output_base_dir="./cv_finetuned_models",
    #     n_splits=10,
    #     test_size=0.1,
    # )

    evaluate_cross_validation(
        data_path="finetune_data.csv",
        model_base_dir="./cv_finetuned_models",
        n_splits=10,
        test_size=0.1,
    )