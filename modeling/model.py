# Copyright (c) 2020, Salesforce.com, Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


from pytorch_transformers.modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, 
                                                 PreTrainedModel, prune_linear_layer, add_start_docstrings)
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel, BertLayer, BertPooler


class BertPointer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPointer, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # classifiers
        self.ext_start_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.ext_end_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.aug_start_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.aug_end_classifier = nn.Linear(config.hidden_size, 1, bias=False)

        self.label_classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, 
                ext_mask=None, ext_start_labels=None, ext_end_labels=None,
                aug_mask=None, aug_start_labels=None, aug_end_labels=None,
                loss_lambda=1.0):
        # run through bert
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)

        # label classifier
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        label_logits = self.label_classifier(pooled_output)

        # extraction classifier
        output = bert_outputs[0]
        ext_mask = ext_mask.unsqueeze(-1)
        ext_start_logits = self.ext_start_classifier(output) * ext_mask
        ext_end_logits = self.ext_end_classifier(output) * ext_mask

        # augmentation classifier
        output = bert_outputs[0]
        aug_mask = aug_mask.unsqueeze(-1)
        aug_start_logits = self.aug_start_classifier(output) * aug_mask
        aug_end_logits = self.aug_end_classifier(output) * aug_mask

        span_logits = (ext_start_logits, ext_end_logits, aug_start_logits, aug_end_logits,)
        outputs = (label_logits,) + span_logits + bert_outputs[2:]

        if labels is not None and \
                ext_start_labels is not None and ext_end_labels is not None and \
                aug_start_labels is not None and aug_end_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(label_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()

                # label loss
                labels_loss = loss_fct(label_logits.view(-1, self.num_labels), labels.view(-1))

                # extraction loss
                ext_start_loss = loss_fct(ext_start_logits.squeeze(), ext_start_labels)
                ext_end_loss = loss_fct(ext_end_logits.squeeze(), ext_end_labels)

                # augmentation loss
                aug_start_loss = loss_fct(aug_start_logits.squeeze(), aug_start_labels)
                aug_end_loss = loss_fct(aug_end_logits.squeeze(), aug_end_labels)

                span_loss = (ext_start_loss + ext_end_loss + aug_start_loss + aug_end_loss) / 4

                # combined loss
                loss = labels_loss + loss_lambda * span_loss

            outputs = (loss, labels_loss, span_loss, ext_start_loss, ext_end_loss, aug_start_loss, aug_end_loss) + outputs

        return outputs  # (loss), (logits), (hidden_states), (attentions)
