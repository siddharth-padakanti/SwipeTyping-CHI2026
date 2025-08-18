import pandas as pd
import torch
from datasets import (Dataset, load_dataset)
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import os
import sys
from datetime import datetime

n_splits = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

def cross_validate_finetune(
    swipe_len: str,
    test_fold: str,
    data_path: str,
):
    fold_dir = "swipe_length_" + str(swipe_len) + "/fold_" + str(test_fold)
    
    prefix = "Find word from trajectory: "
    def preprocess_function(examples):
        """Add prefix to the sentences, tokenize the text, and set the labels"""
        # The "inputs" are the tokenized answer:
        inputs = [prefix + doc for doc in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=32, truncation=True)
        
        # The "labels" are the tokenized outputs:
        labels = tokenizer(text_target=examples["target"], 
                            max_length=8,         
                            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # def preprocess(example):
    #     input_raw  = example.get("input", "").strip()
    #     target_raw = example.get("target", "").strip()
    #     if not input_raw or not target_raw:
    #         return None

    #     input_text  = "Find word from trajectory: " + input_raw
    #     target_text = target_raw

    #     model_inputs = tokenizer(
    #         text=input_text,
    #         max_length=32,
    #         truncation=True,
    #         padding="max_length"
    #     )
    #     labels = tokenizer(
    #         text_target=target_text,
    #         max_length=8,
    #         truncation=True,
    #         padding="max_length"
    #     )
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs

    # 1) Load training CSV via pandas, preserving "null"/"none" as strings
    train_dfs = []
    for fold in range(n_splits):
        if fold != int(test_fold):
            print(fold)
            df = pd.read_csv(
                "swipe_length_" + str(swipe_len) + "/fold_" + str(fold) + "/" + data_path,
                keep_default_na=False,  # don't convert "null"/"none" to NaN
                na_values=[]            # no extra NA tokens
            )
            train_dfs.append(df)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))

    print(train_ds)

    split_dataset = train_ds.train_test_split(test_size=0.2)

    # tokenize
    tokenized_train = split_dataset["train"].map(
        preprocess_function,
        batched=True,
    )

    tokenized_eval = split_dataset["test"].map(
        preprocess_function,
        batched=True,
    )

    # tokenized_train = split_dataset["train"].map(
    #     preprocess,
    #     num_proc=1,
    #     remove_columns=split_dataset["train"].column_names
    # )

    # tokenized_eval = split_dataset["test"].map(
    #     preprocess,
    #     remove_columns=split_dataset["test"].column_names
    # )

    # 1) pick a batch_size variable
    batch_size = 16

    # 2) compute steps_per_epoch before touching training_args
    steps_per_epoch = len(tokenized_train) // batch_size

    print(steps_per_epoch)

    # 3) now build your args
    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        learning_rate=3e-4,
        eval_strategy='steps',
        eval_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        save_total_limit=2,
        output_dir=os.path.join(fold_dir, "checkpoints/"),
        predict_with_generate=False,
        prediction_loss_only=True
    )

    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=fold_dir,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=4,
    #     gradient_accumulation_steps=1,
    #     num_train_epochs=5,
    #     learning_rate=3e-4,
    #     weight_decay=0.01,
    #     logging_dir=os.path.join(fold_dir, "logs"),

    #     # old‐style eval/save flags
    #     do_eval=True,
    #     eval_steps=steps_per_epoch,
    #     logging_steps=steps_per_epoch,
    #     save_steps=steps_per_epoch,

    #     save_total_limit=2,
    #     predict_with_generate=True,
    #     bf16 = True,
    # )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Fold {fold} metrics:", metrics)

    model.save_pretrained(os.path.join(fold_dir, "final/"))
    tokenizer.save_pretrained(os.path.join(fold_dir, "final/"))


    # calculate the accuracy
    test_df = pd.read_csv(
        "swipe_length_" + str(swipe_len) + "/fold_" + str(test_fold) + "/" + data_path,
        keep_default_na=False,  # don't convert "null"/"none" to NaN
        na_values=[]            # no extra NA tokens
    )

    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    result_inputs, result_targets = [], []
    result_counts = []
    result_predicts = []
    result_corrects = []
    result_predicts_top1 = []
    result_predicts_top2 = []
    result_predicts_top3 = []
    result_corrects_top3 = []
    for example in test_ds:
        print(datetime.now())
        print(example)   
        prompt = prefix + example["input"]
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            inputs['input_ids'],
            num_beams=11,
            num_return_sequences=10,
            # max_new_tokens=max_new,
            # min_new_tokens=1,         
            early_stopping=True,
        )
        words = [
            tokenizer.decode(o, skip_special_tokens=True)
            .replace("</s>", "").replace("<pad>", "").replace("<unk>", "").replace(" ", "")
            for o in outputs
        ]
        filtered = [w for w in words if len(w) == int(example["count"])]

        result_inputs.append(example["input"])
        result_targets.append(example["target"])
        result_counts.append(example["count"])

        if len(filtered) == 0:
            result_predicts.append("")
            result_corrects.append(False)
        else:
            result_predicts.append(filtered[0])
            result_corrects.append(filtered[0] == example["target"])

        if len(filtered) == 0:
            result_predicts_top1.append("")
            result_predicts_top2.append("")
            result_predicts_top3.append("")
            result_corrects_top3.append(False)
        elif len(filtered) == 1:
            result_predicts_top1.append(filtered[0])
            result_predicts_top2.append("")
            result_predicts_top3.append("")
            result_corrects_top3.append(filtered[0] == example["target"])
        elif len(filtered) == 2:
            result_predicts_top1.append(filtered[0])
            result_predicts_top2.append(filtered[1])
            result_predicts_top3.append("")
            result_corrects_top3.append(filtered[0] == example["target"] or filtered[1] == example["target"])
        else:
            result_predicts_top1.append(filtered[0])
            result_predicts_top2.append(filtered[1])
            result_predicts_top3.append(filtered[2])
            result_corrects_top3.append(filtered[0] == example["target"] or filtered[1] == example["target"] or filtered[2] == example["target"])


    output_file = fold_dir + "/test_result.csv"
    pd.DataFrame({"input": result_inputs, "target": result_targets, "count": result_counts, "predict": result_predicts, "correct": result_corrects,
        "predicts_top1": result_predicts_top1, "predicts_top2": result_predicts_top2, "predicts_top3": result_predicts_top3, "corrects_top3": result_corrects_top3}).to_csv(output_file, index=False)

def test_finetune(
    swipe_len: str,
    test_fold: str,
    data_path: str,
):
    fold_dir = "swipe_length_" + str(swipe_len) + "/fold_" + str(test_fold)
    prefix = "Find word from trajectory: "

    tokenizer = AutoTokenizer.from_pretrained(str(os.path.join(fold_dir, "final/")), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(str(os.path.join(fold_dir, "final/")), local_files_only=True).to(device)

    # calculate the accuracy
    test_df = pd.read_csv(
        "swipe_length_" + str(swipe_len) + "/fold_" + str(test_fold) + "/" + data_path,
        keep_default_na=False,  # don't convert "null"/"none" to NaN
        na_values=[]            # no extra NA tokens
    )

    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    result_inputs, result_targets = [], []
    result_counts = []
    result_predicts = []
    result_corrects = []
    result_predicts_top1 = []
    result_predicts_top2 = []
    result_predicts_top3 = []
    result_corrects_top3 = []
    for example in test_ds:
        print(datetime.now())
        print(example)        
        prompt = prefix + example["input"]
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            inputs['input_ids'],
            num_beams=11,
            num_return_sequences=10,
            # max_new_tokens=max_new,
            # min_new_tokens=1,         
            early_stopping=True,
        )
        words = [
            tokenizer.decode(o, skip_special_tokens=True)
            .replace("</s>", "").replace("<pad>", "").replace("<unk>", "").replace(" ", "")
            for o in outputs
        ]
        filtered = [w for w in words if len(w) == int(example["count"])]

        result_inputs.append(example["input"])
        result_targets.append(example["target"])
        result_counts.append(example["count"])

        if len(filtered) == 0:
            result_predicts.append("")
            result_corrects.append(False)
        else:
            result_predicts.append(filtered[0])
            result_corrects.append(filtered[0] == example["target"])

        if len(filtered) == 0:
            result_predicts_top1.append("")
            result_predicts_top2.append("")
            result_predicts_top3.append("")
            result_corrects_top3.append(False)
        elif len(filtered) == 1:
            result_predicts_top1.append(filtered[0])
            result_predicts_top2.append("")
            result_predicts_top3.append("")
            result_corrects_top3.append(filtered[0] == example["target"])
        elif len(filtered) == 2:
            result_predicts_top1.append(filtered[0])
            result_predicts_top2.append(filtered[1])
            result_predicts_top3.append("")
            result_corrects_top3.append(filtered[0] == example["target"] or filtered[1] == example["target"])
        else:
            result_predicts_top1.append(filtered[0])
            result_predicts_top2.append(filtered[1])
            result_predicts_top3.append(filtered[2])
            result_corrects_top3.append(filtered[0] == example["target"] or filtered[1] == example["target"] or filtered[2] == example["target"])

    output_file = fold_dir + "/test_result.csv"
    pd.DataFrame({"input": result_inputs, "target": result_targets, "count": result_counts, "predict": result_predicts, "correct": result_corrects,
        "predicts_top1": result_predicts_top1, "predicts_top2": result_predicts_top2, "predicts_top3": result_predicts_top3, "corrects_top3": result_corrects_top3}).to_csv(output_file, index=False)


if __name__ == "__main__":
    
    print(sys.argv)

    swipe_length = sys.argv[1]
    fold = sys.argv[2]

    print("finetune model for swipe length " + swipe_length + ", fold " + fold)

    fold_dir = "./swipe_length_" + str(swipe_length) + "/fold_" + str(fold)
    
    cross_validate_finetune(
        swipe_len = swipe_length,
        test_fold = fold,
        data_path="finetune_data.csv",
    )

    # test_finetune(
    #     swipe_len = swipe_length,
    #     test_fold = fold,
    #     data_path="finetune_data.csv",
    # )