import pandas as pd
import pytorch_lightning as pl

from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from model import SpellcheckerDataModule, SoftMaskedBert, MLMBert
from evaluation import show_model_performance

import os
import torch
import argparse


def main(MODEL, BERT, DEVICE):

    BATCH_SIZE = 32
    MAX_TOKEN_LEN = 64
    NUM_WORKERS = 16

    TRAIN_DATA_PATH = "/home/jovyan/sqqqqaid/spellchecker_data/statistically_augmented_dataset_500k.txt"
    EVAL_DATA_PATH = "/home/jovyan/sqqqqaid/spellchecker_data/val_set.txt"
    CHECKPOINTS_PATH = "/home/jovyan/sqqqqaid/spellchecker_checkpoints"


    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('high')

    print("[INITIALIZING MODEL...]")
    tokenizer = AutoTokenizer.from_pretrained(BERT)
    if MODEL == "masked":
        model = MLMBert(BERT)
    elif MODEL == "softmasked":
        model = SoftMaskedBert(BERT, mask_token_id=tokenizer.mask_token_id)
    print("[COMPLETE INITIALIZING MODEL...]")

    print("[READING DATA...]")
    data = pd.read_csv(TRAIN_DATA_PATH, delimiter="\t", header=0)
    predict_data = pd.read_csv(EVAL_DATA_PATH, delimiter="\t", header=0)
    print("COMPLETE READING DATA]")

    data_module = SpellcheckerDataModule(
        data,
        tokenizer,
        MODEL,
        predict_data=predict_data,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_LEN,
        num_workers=NUM_WORKERS,

    )

    wandb_logger = WandbLogger(project="spellchecker")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=CHECKPOINTS_PATH,
                                                       filename="version0",
                                                       save_top_k=-1,
                                                       verbose=True,
                                                       monitor="val_loss",
                                                       mode="min")

    trainer = Trainer(
        accelerator=DEVICE,
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=20),
        ],
        # fast_dev_run=True,
        log_every_n_steps=50,
        val_check_interval=0.1,
    )

    trainer.fit(model, data_module)

    # Caculate model's performance
    # metrics = show_model_performance(model, tokenizer, predict_data)
    # print(metrics)

    print("Completed")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="softmasked")  # softmasked or masked
    argparser.add_argument("--bert", type=str, default="bert-base-multilingual-cased")
    argparser.add_argument("--device", type=str, default="gpu")
    args = argparser.parse_args()

    main(args.model, args.bert, args.device)