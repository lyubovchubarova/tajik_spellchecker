import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from pytorch_lightning import LightningDataModule


class SpellcheckerDataset(Dataset):
    def __init__(self, data, tokenizer, model_type, max_length=32):
        """
        :param data: DataFrame with ["mistaken", "corrected"] columns.
        :param tokenizer: transformers tokenizer from pretrained.
        :param model_type: "masked" or "softmasked".
        :param max_length: max len to pad or truncate inputs to.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.model_type = model_type
        self.max_length = max_length
        self.mistaken, self.corrected = list(self.data.columns)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: dict fo encodings, where
             input_ids are bert tokenized string with mistakes,
            * output_ids are bert tokenized corrected string,
            * labels are {0,1} indicating if the token contains a mistake.
        """
        encodings = self.tokenizer.encode_plus(
            self.data.iloc[idx][self.mistaken],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        encodings["output_ids"] = self.tokenizer.encode_plus(
            self.data.iloc[idx][self.corrected],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )["input_ids"]

        encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        encodings["labels"] = (encodings["input_ids"] == encodings["output_ids"])

        if self.model_type == "masked":
            encodings["labels"] = encodings["labels"].long()
        elif self.model_type == "softmasked":
            encodings["labels"] = encodings["labels"].float()

        return encodings

    def __len__(self):
        return len(self.data)


class SpellcheckerDataModule(LightningDataModule):

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 model_type: str,  # "masked" or "softmasked"
                 predict_data: pd.DataFrame = pd.DataFrame(),
                 batch_size: int = 32,
                 eval_fraction=0.3,
                 max_token_len: int = 128,
                 num_workers=16,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_type = model_type

        self.data = data
        self.predict_data = predict_data

        self.batch_size = batch_size
        self.eval_fraction = eval_fraction

        self.max_token_len = max_token_len
        self.num_workers = num_workers

    def setup(self, stage: str):
        data_full = SpellcheckerDataset(self.data,
                                        self.tokenizer,
                                        self.model_type,
                                        self.max_token_len)

        eval_len = int(self.data.shape[0] * self.eval_fraction)
        train_len = self.data.shape[0] - eval_len

        self.data_train, self.data_eval = random_split(
            data_full,
            [train_len, eval_len],
            generator=torch.Generator().manual_seed(1000)
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.data_eval, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        dataset = SpellcheckerDataset(self.predict_data, self.tokenizer, self.model_type, self.max_token_len)
        return DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)

