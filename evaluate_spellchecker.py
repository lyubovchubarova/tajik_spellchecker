import pandas as pd
import pytorch_lightning as pl

from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from model import SpellcheckerDataModule, SoftMaskedBert, MLMBert
from evaluation import compute_metrics

import os
import torch
import argparse


def main(MODEL, BERT, CHECKPOINT):
    # if not CHECKPOINT:
    #     raise

    MAX_TOKEN_LEN = 64
    NUM_WORKERS = 16

    EVAL_DATA_PATH = "/home/jovyan/sqqqqaid/spellchecker_data/val_set.txt"
    EVAL_DATA_PATH = "data/spellchecker_data/val_set.txt"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('high')

    print("[INITIALIZING MODEL...]")
    tokenizer = AutoTokenizer.from_pretrained(BERT)
    if MODEL == "masked":
        model = MLMBert(BERT)
    elif MODEL == "softmasked":
        model = SoftMaskedBert(BERT, mask_token_id=tokenizer.mask_token_id)
    print("[COMPLETE INITIALIZING MODEL]")

    # print(f"[LOADING MODEL FROM CHECKPOINT {CHECKPOINT}...]")
    # if MODEL == "masked":
    #     loading_kwargs = {"model_name": BERT}
    # elif MODEL == "softmasked":
    #     loading_kwargs = {"model_name": BERT, "mask_token_id": tokenizer.mask_token_id}
    # model.load_from_checkpoint(CHECKPOINT, **loading_kwargs)
    # print("[COMPLETE LOADING MODEL FROM CHECKPOINT]")

    print("[READING DATA...]")
    predict_data = pd.read_csv(EVAL_DATA_PATH, delimiter="\t", header=0)
    data_module = SpellcheckerDataModule(
        pd.DataFrame(),
        tokenizer,
        MODEL,
        predict_data=predict_data[:2],
        batch_size=1,
        max_token_len=MAX_TOKEN_LEN,
        num_workers=NUM_WORKERS,

    )
    print("[COMPLETE READING DATA]")

    print("[CALCULATING METRICS...]")
    metrics = compute_metrics(model, data_module)
    print(metrics)

    print("COMPLETED")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="softmasked")  # softmasked or masked
    argparser.add_argument("--bert", type=str, default="bert-base-multilingual-cased")
    argparser.add_argument("--checkpoint_path", type=str, default="")
    args = argparser.parse_args()

    main(args.model, args.bert, args.checkpoint_path)