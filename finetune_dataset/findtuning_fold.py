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

n_splits = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.memory_summary(device=None, abbreviated=False)) 


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl").to(device)

def cross_validate_finetune(
    test_fold: int,
    data_path: str,
):
    fold_dir = "fold_" + str(test_fold)
    
    tokenizer_fold = tokenizer
    model_fold = model

    def preprocess(example):
        input_raw  = example.get("input", "").strip()
        target_raw = example.get("target", "").strip()
        if not input_raw or not target_raw:
            return None

        input_text  = "Find word from trajectory: " + input_raw
        target_text = target_raw

        model_inputs = tokenizer_fold(
            text=input_text,
            max_length=32,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer_fold(
            text_target=target_text,
            max_length=8,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 1) Load training CSV via pandas, preserving "null"/"none" as strings
    train_dfs = []
    for fold in range(n_splits):
        if fold != test_fold:
            df = pd.read_csv(
                "fold_" + str(fold) + "/" + data_path,
                keep_default_na=False,  # don't convert "null"/"none" to NaN
                na_values=[]            # no extra NA tokens
            )
            train_dfs.append(df)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))

    split_dataset = train_ds.train_test_split(test_size=0.2)

    # tokenize
    tokenized_train = split_dataset["train"].map(
        preprocess,
        num_proc=1,
        remove_columns=split_dataset["train"].column_names
    )

    tokenized_eval = split_dataset["test"].map(
        preprocess,
        remove_columns=split_dataset["test"].column_names
    )

    # 1) pick a batch_size variable
    batch_size = 8

    # 2) compute steps_per_epoch before touching training_args
    steps_per_epoch = len(tokenized_train) // batch_size

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
        model=model_fold,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer_fold,
        data_collator=DataCollatorForSeq2Seq(tokenizer_fold, model=model_fold),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Fold {fold+1} metrics:", metrics)

    model_fold.save_pretrained(fold_dir)
    tokenizer_fold.save_pretrained(fold_dir)


if __name__ == "__main__":

    for fold in range(n_splits):
        cross_validate_finetune(
            test_fold = fold,
            data_path="finetune_data.csv",
        )