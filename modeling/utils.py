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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import json
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 extraction_span=None, augmentation_span=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.extraction_span = extraction_span
        self.augmentation_span = augmentation_span


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 extraction_mask=None, extraction_start_ids=None, extraction_end_ids=None,
                 augmentation_mask=None, augmentation_start_ids=None, augmentation_end_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.extraction_mask = extraction_mask
        self.extraction_start_ids = extraction_start_ids
        self.extraction_end_ids = extraction_end_ids
        self.augmentation_mask = augmentation_mask
        self.augmentation_start_ids = augmentation_start_ids
        self.augmentation_end_ids = augmentation_end_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a jsonl file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
        return lines


class FactCCGeneratedProcessor(DataProcessor):
    """Processor for the generated FactCC data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["CORRECT", "INCORRECT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in lines:
            guid = example["id"]
            text_a = example["text"]
            text_b = example["claim"]
            label = example["label"]
            extraction_span = example["extraction_span"]
            augmentation_span = example["augmentation_span"]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             extraction_span=extraction_span, augmentation_span=augmentation_span))
        return examples


class FactCCManualProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "data-dev.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["CORRECT", "INCORRECT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, example) in enumerate(lines):
            guid = str(i)
            text_a = example["text"]
            text_b = example["claim"]
            label = example["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        ####### AUX LOSS DATA
        # get tokens_a mask
        extraction_span_len = len(tokens_a) + 2
        extraction_mask = [1 if 0 < ix < extraction_span_len else 0 for ix in range(max_seq_length)]

        # get extraction labels
        if example.extraction_span:
            ext_start, ext_end = example.extraction_span
            extraction_start_ids = ext_start + 1
            extraction_end_ids = ext_end + 1
        else:
            extraction_start_ids = extraction_span_len
            extraction_end_ids = extraction_span_len

        augmentation_mask = [1 if extraction_span_len <= ix < extraction_span_len + len(tokens_b) + 1  else 0 for ix in range(max_seq_length)]

        if example.augmentation_span:
            aug_start, aug_end = example.augmentation_span
            augmentation_start_ids = extraction_span_len + aug_start
            augmentation_end_ids = extraction_span_len + aug_end
        else:
            last_sep_token = extraction_span_len + len(tokens_b)
            augmentation_start_ids = last_sep_token
            augmentation_end_ids = last_sep_token

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("ext mask: %s" % " ".join([str(x) for x in extraction_mask]))
            logger.info("ext start: %d" % extraction_start_ids)
            logger.info("ext end: %d" % extraction_end_ids)
            logger.info("aug mask: %s" % " ".join([str(x) for x in augmentation_mask]))
            logger.info("aug start: %d" % augmentation_start_ids)
            logger.info("aug end: %d" % augmentation_end_ids)
            logger.info("label: %d" % label_id)

        extraction_start_ids = min(extraction_start_ids, 511)
        extraction_end_ids = min(extraction_end_ids, 511)
        augmentation_start_ids = min(augmentation_start_ids, 511)
        augmentation_end_ids = min(augmentation_end_ids, 511)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          extraction_mask=extraction_mask,
                          extraction_start_ids=extraction_start_ids,
                          extraction_end_ids=extraction_end_ids,
                          augmentation_mask=augmentation_mask,
                          augmentation_start_ids=augmentation_start_ids,
                          augmentation_end_ids=augmentation_end_ids))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def complex_metric(preds, labels, prefix=""):
    return {
        prefix + "bacc": balanced_accuracy_score(y_true=labels, y_pred=preds),
        prefix + "f1":   f1_score(y_true=labels, y_pred=preds, average="micro")
    }


def compute_metrics(task_name, preds, labels, prefix=""):
    assert len(preds) == len(labels)
    if task_name == "factcc_generated":
        return complex_metric(preds, labels, prefix)
    elif task_name == "factcc_annotated":
        return complex_metric(preds, labels, prefix)
    else:
        raise KeyError(task_name)

processors = {
    "factcc_generated": FactCCGeneratedProcessor,
    "factcc_annotated": FactCCManualProcessor,
}

output_modes = {
    "factcc_generated": "classification",
    "factcc_annotated": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "factcc_generated": 2,
    "factcc_annotated": 2,
}
