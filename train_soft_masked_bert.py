import pandas as pd
import pytorch_lightning as pl

from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from model import SpellcheckerDataModule, SoftMaskedBert
from evaluation import show_model_performance

import os
import torch


def main():
    BATCH_SIZE = 32
    MAX_TOKEN_LEN = 64
    NUM_WORKERS = 16
    TRAIN_DATA_PATH = "/home/jovyan/sqqqqaid/spellchecker_data/statistically_augmented_dataset_500k.txt"
    EVAL_DATA_PATH = "/home/jovyan/sqqqqaid/spellchecker_data/val_set.txt"
    CHECKPOINTS_PATH = "/home/jovyan/sqqqqaid/spellchecker_checkpoints"


    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('high')

    print("[INITIALIZING MODEL...]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = SoftMaskedBert("bert-base-multilingual-cased", mask_token_id=tokenizer.mask_token_id)
    print("[COMPLETE INITIALIZING MODEL...]")

    print("[READING DATA...]")
    data = pd.read_csv(TRAIN_DATA_PATH, delimiter="\t", header=0)
    predict_data = pd.read_csv(EVAL_DATA_PATH, delimiter="\t", header=0)
    print("COMPLETE READING DATA]")

    data_module = SpellcheckerDataModule(
        data,
        tokenizer,
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
        accelerator="gpu",
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
    main()