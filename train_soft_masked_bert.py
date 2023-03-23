import pandas as pd
import pytorch_lightning as pl

from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import SpellcheckerDataModule, SoftMaskedBert
from evaluation import show_model_performance


def main():
    BATCH_SIZE = 2
    MAX_TOKEN_LEN = 64

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = SoftMaskedBert("bert-base-multilingual-cased", mask_token_id=tokenizer.mask_token_id)

    print("reading data...", end=" ")
    #data = pd.read_csv("data/statistically_augmented_dataset.txt", delimiter="\t", header=0)[:10]
    data = pd.read_csv("data/val_set.txt", delimiter="\t", header=0)[5:15]
    predict_data = pd.read_csv("data/val_set.txt", delimiter="\t", header=0)[:5]
    print("complete")
    data_module = SpellcheckerDataModule(
        data,
        tokenizer,
        predict_data=predict_data,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_LEN
    )

    #logger = TensorBoardLogger(save_dir="tb_logs", name="TajikSpellchecker")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints",
                                                       filename="version_7",
                                                       save_top_k=1,
                                                       verbose=True,
                                                       monitor="val_loss",
                                                       mode="min")

    trainer = Trainer(
        max_epochs=10,
        #logger=logger,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=20),
            ],
        fast_dev_run=True,
        log_every_n_steps=50
    )

    trainer.fit(model, data_module)

    # Caculate model's performance
    metrics = show_model_performance(model, tokenizer, predict_data)
    print(metrics)

    print("Completed")

if __name__ == "__main__":
    main()