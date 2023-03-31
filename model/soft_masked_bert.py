import torch
import torch.nn as nn
from typing import Optional

from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from .train_config import TrainConfig


class SoftMaskedBert(TrainConfig):

    def __init__(self, model_name: str, mask_token_id: int):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertForMaskedLM.from_pretrained(model_name)

        self.mask_token_id = mask_token_id
        self.vocab_size = self.bert.config.vocab_size

        # Word embedding
        self.embeddings = self.bert.bert.embeddings

        # detection
        self.bidirectional_gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.bert.config.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.linear = nn.Linear(self.bert.config.hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

        # correction
        self.encoder = self.bert.bert.encoder
        self.cls = self.bert.cls

        # Loss function
        self.det_criterion = nn.BCELoss()
        self.cor_criterion = nn.CrossEntropyLoss()

        # coef to compute loss
        self.correction_coef = 0.8

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ):
        embeddings = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        # detection
        detection_probs = self._get_detection_probs(embeddings)

        # soft masking
        soft_masked_embeddings, extended_attention_mask = self._soft_mask(
            embeddings, detection_probs, attention_mask, input_ids,
        )

        # correction
        corrections_logits = self._get_corrections_logits(embeddings, soft_masked_embeddings, extended_attention_mask)

        loss = None
        if output_ids is not None and labels is not None:
            det_loss = self.det_criterion(detection_probs.squeeze(), labels)
            cor_loss = self.cor_criterion(corrections_logits.view(-1, self.vocab_size), output_ids.view(-1))
            loss = self.correction_coef * cor_loss + (1 - self.correction_coef) * det_loss

        return MaskedLMOutput(
            loss=loss,
            logits=corrections_logits,
        )

    def _get_detection_probs(self, embeddings):
        gru_output, _ = self.bidirectional_gru(embeddings)
        probs = self.sigmoid((self.linear(gru_output)))
        return probs

    def _soft_mask(
            self,
            embeddings,
            detection_probs,
            attention_mask,
            input_ids,
    ):
        masked_e = self.embeddings(torch.tensor([[self.mask_token_id]], dtype=torch.long).to(self.device))
        soft_masked_embeddings = detection_probs * masked_e + (1 - detection_probs) * embeddings

        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L852
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(
            attention_mask,
            input_ids.size()
        )
        return soft_masked_embeddings, extended_attention_mask

    def _get_corrections_logits(self, embeddings, soft_masked_embeddings, extended_attention_mask):
        bert_out = self.encoder(hidden_states=soft_masked_embeddings,
                                attention_mask=extended_attention_mask)
        h = bert_out[0] + embeddings
        corrections_logits = self.cls(h)

        return corrections_logits

