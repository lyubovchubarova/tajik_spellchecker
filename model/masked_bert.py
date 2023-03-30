import torch

from typing import Optional
from transformers import BertForMaskedLM

from .train_config import TrainConfig


class MLMBert(TrainConfig):
        
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(model_name)
        
    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_ids: Optional[torch.Tensor] = None,  # aren't use, here to unify training
        ):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels.type(torch.LongTensor),
        )
    