#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.metrics import precision_recall_fscore_support

def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy + stats on finetune_data.csv"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Dir with config.json, pytorch_model.bin, tokenizer, etc."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="finetune_data.csv",
        help="Path to finetune_data.csv"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for generation"
    )
    args = parser.parse_args()

    # 1) Load data
    df = pd.read_csv(args.csv_path, keep_default_na=False, na_values=[])
    total = len(df)

    # 2) Load model & tokenizer
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model     = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    # 3) Inference loop
    y_true, y_pred = [], []
    for _, row in tqdm(df.iterrows(), total=total, desc="Evaluating"):
        traj   = row["input"]
        target = str(row["target"]).strip()
        prompt = f"Find word from trajectory: {traj}"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outs   = model.generate(
            inputs["input_ids"],
            max_length=8,
            num_beams=args.beam_size,
            early_stopping=True
        )
        pred = tokenizer.decode(outs[0], skip_special_tokens=True).strip()
        y_true.append(target)
        y_pred.append(pred)

    # 4) Overall accuracy
    correct  = sum(1 for t,p in zip(y_true,y_pred) if t==p)
    accuracy = correct / total
    print(f"\nOverall accuracy: {accuracy:.4f} ({correct}/{total})")

    # 5) Precision / Recall / F1 (weighted over all words)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall:    {recall:.4f}")
    print(f"Weighted F1:        {f1:.4f}")

    # 6) Accuracy by word-length
    df_stats = pd.DataFrame({"true": y_true, "pred": y_pred})
    df_stats["length"] = df_stats["true"].str.len()
    by_len = (
        df_stats
        .assign(correct=lambda d: d["true"]==d["pred"])
        .groupby("length")["correct"]
        .agg(["mean","count"])
        .rename(columns={"mean":"accuracy","count":"examples"})
    )
    print("\nAccuracy by target-word length:")
    print(by_len.to_string(float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
