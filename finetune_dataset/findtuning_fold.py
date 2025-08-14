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


n_splits = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

def cross_validate_finetune(
    test_fold: int,
    data_path: str,
):
    fold_dir = "fold_" + str(test_fold)
    
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
                "fold_" + str(fold) + "/" + data_path,
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

    model.save_pretrained(fold_dir)
    tokenizer.save_pretrained(fold_dir)


if __name__ == "__main__":
    
    print(sys.argv)

    fold = sys.argv[1]

    print("finetune model for fold " + fold)

    fold_dir = "./fold_" + str(fold)

    output_dir = os.path.join(fold_dir, "checkpoints/")

    print(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    cross_validate_finetune(
        test_fold = fold,
        data_path="finetune_data.csv",
    )