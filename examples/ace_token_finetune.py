# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import os
import logging
import argparse
import random
import datetime
from tqdm import tqdm, trange
from pathlib import Path
import math
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import printable_text, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForTokenClassification,BertCRF
from pytorch_pretrained_bert.optimization import BertAdam

from conlleval import evaluate
from conlleval import evaluate2

logging.basicConfig(filename = '{}_log.txt'.format(datetime.datetime.now()),
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

logger.info(PYTORCH_PRETRAINED_BERT_CACHE)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, labels=None):
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
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


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
    def _read_tsv(cls, input_file):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        lines = [line.strip().split("\t") for line in lines]
        return lines

class TriggerProcessor(DataProcessor):
    """Processor for the Hyperpartisan data set."""

    def __init__(self, etype):
        self.etype = etype

    def get_examples(self, input_file, test=False):
        """See base class."""
        logger.info("LOOKING AT {}".format(input_file))
        if test:
            return self._create_examples_for_test(self._read_tsv(input_file))
        return self._create_examples(self._read_tsv(input_file))

    def get_labels(self):
        """See base class."""
        if self.etype == "Business":
            return ["B-Start-Org", "B-End-Org", "B-Declare-Bankruptcy", "B-Merge-Org", "I-Start-Org", "I-End-Org", "I-Declare-Bankruptcy", "I-Merge-Org", "O"]
        elif self.etype == "Conflict":
            return ["B-Attack", "B-Demonstrate", "I-Attack", "I-Demonstrate", "O"]
        elif self.etype == "Contact":
            return ["B-Meet", "B-Phone-Write", "I-Meet", "I-Phone-Write", "O"]
        elif self.etype == "Justice":
            return ["B-Arrest-Jail", "B-Release-Parole", "B-Charge-Indict", "B-Trial-Hearing", "B-Sue", "B-Convict", "B-Sentence", "B-Fine", "B-Execute", "B-Extradite", "B-Acquit", "B-Pardon", "B-Appeal", "I-Arrest-Jail", "I-Release-Parole", "I-Charge-Indict", "I-Trial-Hearing", "I-Sue", "I-Convict", "I-Sentence", "I-Fine", "I-Execute", "I-Extradite", "I-Acquit", "I-Pardon", "I-Appeal", "O"]
        elif self.etype == "Life":
            return ["B-Be-Born", "B-Die", "B-Marry", "B-Divorce", "B-Injure", "I-Be-Born", "I-Die", "I-Marry", "I-Divorce", "I-Injure", "O"]
        elif self.etype == "Movement":
            return ["B-Transport", "I-Transport", "O"]
        elif self.etype == "Personnel":
            return ["B-Start-Position", "B-End-Position", "B-Nominate", "B-Elect", "I-Start-Position", "I-End-Position", "I-Nominate", "I-Elect", "O"]
        elif self.etype == "Transaction":
            return ["B-Transfer-Ownership", "B-Transfer-Money", "I-Transfer-Ownership", "I-Transfer-Money", "O"]
        else:
            raise Exception("Etype %s not known!" %self.etype)

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
#        self.label_list = lines.label.unique().tolist()
        examples = []
        words = []
        labels = []
        j = 0
        for (i, line) in enumerate(lines):
            guid = j
            if line[0] == "SAMPLE_START":
                words.append("[CLS]")
                labels.append(-1)
            elif line[0] == "[SEP]":
                # Since we may have more than two sentences in a sample, we can not assign the third sentences tokens segment ids
                # words.append("[PAD]")
                # labels.append(-1)
                continue
            elif line[0] == "":
                examples.append(InputExample(guid=guid, text_a=words, labels=labels))
                j += 1
                words = []
                labels = []
                continue
            elif line[0] in ["\x91", "\x92", "\x97"]:
                continue
            else:
                words.append(line[0])
                labels.append(line[1])

        return examples

    def _create_examples_for_test(self, lines):
        """Creates examples for the training and dev sets."""
#        self.label_list = lines.label.unique().tolist()
        examples = []
        words = []
        j = 0
        for (i, line) in enumerate(lines):
            guid = j
            if line[0] == "SAMPLE_START":
                words.append("[CLS]")
            elif line[0] == "[SEP]":
                # Since we may have more than two sentences in a sample, we can not assign the third sentences tokens segment ids
                # words.append("[PAD]")
                # labels.append(-1)
                continue
            elif line[0] == "":
                examples.append(InputExample(guid=guid, text_a=words))
                j += 1
                words = []
                continue
            else:
                words.append(line[0])

        return examples

class AllProcessor(DataProcessor):
    """Processor for the Hyperpartisan data set."""

    def __init__(self, etype):
        self.etype = etype

    def get_examples(self, input_file, test=False):
        """See base class."""
        logger.info("LOOKING AT {}".format(input_file))
        if test:
            return self._create_examples_for_test(self._read_tsv(input_file))
        return self._create_examples(self._read_tsv(input_file))

    #["Person", "Place", "Buyer", "Seller", "Beneficiary", "Price", "Artifact", "Origin", "Destination", "Giver", "Recipient", "Money", "Org", "Agent", "Victim", "Instrument", "Entity", "Attacker", "Target", "Defendant", "Adjudicator", "Prosecutor", "Plaintiff", "Crime", "Position", "Sentence", "Vehicle", "Time-Within", "Time-Starting", "Time-Ending", "Time-Before", "Time-After", "Time-Holds", "Time-At-Beginning", "Time-At-End"]

    # TODO : Maybe some event types only use a subset of these arguments
    def get_labels(self):
        """See base class."""
        if self.etype == "Business":
            return ["B-Start-Org", "B-End-Org", "B-Declare-Bankruptcy", "B-Merge-Org", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Start-Org", "I-End-Org", "I-Declare-Bankruptcy", "I-Merge-Org", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        elif self.etype == "Conflict":
            return ["B-Attack", "B-Demonstrate", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Attack", "I-Demonstrate", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        elif self.etype == "Contact":
            return ["B-Meet", "B-Phone-Write", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Meet", "I-Phone-Write", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        elif self.etype == "Justice":
            return ["B-Arrest-Jail", "B-Release-Parole", "B-Charge-Indict", "B-Trial-Hearing", "B-Sue", "B-Convict", "B-Sentence", "B-Fine", "B-Execute", "B-Extradite", "B-Acquit", "B-Pardon", "B-Appeal", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Arrest-Jail", "I-Release-Parole", "I-Charge-Indict", "I-Trial-Hearing", "I-Sue", "I-Convict", "I-Sentence", "I-Fine", "I-Execute", "I-Extradite", "I-Acquit", "I-Pardon", "I-Appeal", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        elif self.etype == "Life":
            return ["B-Be-Born", "B-Die", "B-Marry", "B-Divorce", "B-Injure", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Be-Born", "I-Die", "I-Marry", "I-Divorce", "I-Injure", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        elif self.etype == "Movement":
            return ["B-Transport", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Transport", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        elif self.etype == "Personnel":
            return ["B-Start-Position", "B-End-Position", "B-Nominate", "B-Elect", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Start-Position", "I-End-Position", "I-Nominate", "I-Elect", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        elif self.etype == "Transaction":
            return ["B-Transfer-Ownership", "B-Transfer-Money", "B-Person", "B-Place", "B-Buyer", "B-Seller", "B-Beneficiary", "B-Price", "B-Artifact", "B-Origin", "B-Destination", "B-Giver", "B-Recipient", "B-Money", "B-Org", "B-Agent", "B-Victim", "B-Instrument", "B-Entity", "B-Attacker", "B-Target", "B-Defendant", "B-Adjudicator", "B-Prosecutor", "B-Plaintiff", "B-Crime", "B-Position", "B-Sentence", "B-Vehicle", "B-Time-Within", "B-Time-Starting", "B-Time-Ending", "B-Time-Before", "B-Time-After", "B-Time-Holds", "B-Time-At-Beginning", "B-Time-At-End", "I-Transfer-Ownership", "I-Transfer-Money", "I-Person", "I-Place", "I-Buyer", "I-Seller", "I-Beneficiary", "I-Price", "I-Artifact", "I-Origin", "I-Destination", "I-Giver", "I-Recipient", "I-Money", "I-Org", "I-Agent", "I-Victim", "I-Instrument", "I-Entity", "I-Attacker", "I-Target", "I-Defendant", "I-Adjudicator", "I-Prosecutor", "I-Plaintiff", "I-Crime", "I-Position", "I-Sentence", "I-Vehicle", "I-Time-Within", "I-Time-Starting", "I-Time-Ending", "I-Time-Before", "I-Time-After", "I-Time-Holds", "I-Time-At-Beginning", "I-Time-At-End", "O"]
        else:
            raise Exception("Etype %s not known!" %self.etype)

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
#        self.label_list = lines.label.unique().tolist()
        examples = []
        words = []
        labels = []
        j = 0
        for (i, line) in enumerate(lines):
            guid = j
            if line[0] == "SAMPLE_START":
                words.append("[CLS]")
                labels.append(-1)
            elif line[0] == "[SEP]":
                # Since we may have more than two sentences in a sample, we can not assign the third sentences tokens segment ids
                # words.append("[PAD]")
                # labels.append(-1)
                continue
            elif line[0] == "":
                examples.append(InputExample(guid=guid, text_a=words, labels=labels))
                j += 1
                words = []
                labels = []
                continue
            elif line[0] in ["\x91", "\x92", "\x97"]:
                continue
            else:
                words.append(line[0])
                labels.append(line[1])

        return examples

    def _create_examples_for_test(self, lines):
        """Creates examples for the training and dev sets."""
#        self.label_list = lines.label.unique().tolist()
        examples = []
        words = []
        j = 0
        for (i, line) in enumerate(lines):
            guid = j
            if line[0] == "SAMPLE_START":
                words.append("[CLS]")
            elif line[0] == "[SEP]":
                # Since we may have more than two sentences in a sample, we can not assign the third sentences tokens segment ids
                # words.append("[PAD]")
                # labels.append(-1)
                continue
            elif line[0] == "":
                examples.append(InputExample(guid=guid, text_a=words))
                j += 1
                words = []
                continue
            else:
                words.append(line[0])

        return examples

def convert_examples_to_features(example, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens = []
    label_ids = []
    # words = example.text_a

    for (i, word) in enumerate(example.text_a):
        # if word == "[SEP]":
        #     continue
        if word == "[CLS]":
            tokens.append(word)
            continue

        # logger.info(word)
        tokenized = tokenizer.tokenize(word)
        # logger.info(tokenized)
        # logger.info("--------")
        # If we want to keep all wordpieces
        # tokens.extend(tokenized)
        # label_ids.extend(len(tokenized)*[label_map[example.labels[i]]])
        try:
            tokens.append(tokenized[0])
        except:
            tokens.append("[UNK]")
        # label_ids.append(label_map[example.labels[i]])

    # label_ids = [label_map[x] if x not in [-1, -2] else x for x in example.labels]
    label_ids = [label_map[x] if x != -1 else x for x in example.labels]

    if len(tokens) > max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
        # words = example.text_a[0:(max_seq_length - 1)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    # We insert in case that length was bigger than max_seq_length
    tokens.append("[SEP]")
    label_ids.append(-1)

    segment_ids = [0] * len(tokens)

#    tokens = [token for token in tokens if token in tokenizer.vocab.keys() else "[UNK]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=label_ids)


class HyperpartisanData(Dataset):
    """"""
    def __init__(self, examples, label_list, max_seq_length, tokenizer):
        self.examples = examples
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        feats = convert_examples_to_features(ex, self.label_list, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor(feats.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feats.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feats.segment_ids, dtype=torch.long)
        label_ids = torch.tensor(feats.label_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, label_ids


def convert_examples_to_features_for_test(example, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    tokens = []
    # words = example.text_a

    for (i, word) in enumerate(example.text_a):
        # if word == "[SEP]":
        #     continue
        if word == "[CLS]":
            tokens.append(word)
            continue

        tokenized = tokenizer.tokenize(word)
        try:
            tokens.append(tokenized[0])
        except:
            tokens.append("[UNK]")

    if len(tokens) > max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]

    tokens.append("[SEP]")
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids)


class HyperpartisanData_for_test(Dataset):
    """"""
    def __init__(self, examples, max_seq_length, tokenizer):
        self.examples = examples
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        feats = convert_examples_to_features_for_test(ex, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor(feats.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feats.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feats.segment_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, ex.guid

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

def accuracy(outputs, labels):
    return np.sum(outputs == labels)/len(labels)

def get_rates(out, labels):
    outputs = np.argmax(out, axis=1)
    adder = outputs + labels
    TP = len(adder[adder == 2])
    TN = len(adder[adder == 0])
    subtr = labels - outputs
    FP = len(subtr[subtr == -1])
    FN = len(subtr[subtr == 1])

    return np.array([TP, TN, FP, FN])

def get_scores(rates):

    [TP, TN, FP, FN] = rates

    balanced_acc = ((TP / (TP+FN)) + (TN / (TN+FP))) / 2
    mcc = (TP*TN - FP*FN) / math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

    precision_2 = TP / (TP + FP)
    precision_1 = TN / (TN + FN)
    recall_2 = TP / (TP + FN)
    recall_1 = TN / (TN + FP)
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    f1_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2)

    return balanced_acc, f1_1, f1_2, mcc, recall_2, precision_2


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--etype",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of event type.")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--bert_tokenizer", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_file",
                        default="",
                        type=str,
                        help="The path of train file.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--dev_file",
                        default="",
                        type=str,
                        help="The path of eval file.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--test_file",
                        default="",
                        type=str,
                        help="The path of test file.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--model_load",
                        default="",
                        type=str,
                        help="The path of model state.")
    parser.add_argument("--val_each",
                        default=500000,
                        type=int,
                        help="every nth iteration to do eval")
    parser.add_argument('--use_crf',
                        default=False,
                        action='store_true',
                        help="Whether to use CRF layer at the end.")


    args = parser.parse_args()

    processors = {
        "trigger": TriggerProcessor,
        "all": AllProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if not args.do_train and not args.do_eval and not args.do_test:
    #     raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    #os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](args.etype)

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

    label_list = processor.get_labels()
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[i] = label


    etype_label_list = ["Life", "Transaction", "Movement", "Business", "Conflict", "Contact", "Personnel", "Justice"]
    etype_label_map = {}
    for i,label in enumerate(etype_label_list):
        etype_label_map[i] = label


    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_examples(args.train_file)
        random.shuffle(train_examples)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        train_dataloader = DataLoader(dataset=HyperpartisanData(train_examples, label_list, args.max_seq_length, tokenizer), batch_size=args.train_batch_size)

    if args.do_eval:
        eval_examples = processor.get_examples(args.dev_file, test=True)
        eval_dataloader = DataLoader(dataset=HyperpartisanData_for_test(eval_examples, args.max_seq_length, tokenizer), batch_size=args.train_batch_size)
        dev_df = pd.read_json("/scratch/users/omutlu/ace2005/Token/dev.json", lines=True, orient="records")
        dev_df.labels = dev_df.labels.apply(lambda x: [etype_label_map[label] for label in x])

    if args.do_test:
        test_examples = processor.get_examples(args.test_file, test=True)
        test_dataloader = DataLoader(dataset=HyperpartisanData_for_test(test_examples, args.max_seq_length, tokenizer), batch_size=args.train_batch_size)
        test_df = pd.read_json("/scratch/users/omutlu/ace2005/Token/test.json", lines=True, orient="records")
        test_df.labels = test_df.labels.apply(lambda x: [etype_label_map[label] for label in x])

    # Prepare model
    num_labels = len(label_list)
    constraints = [] # (from,to)
    if args.use_crf:
        I_list = [i for i,v in enumerate(label_list) if "I-" in v]
        for i,v in enumerate(I_list): # For all "I-"s constraint all other "I-"s
            constraints.extend([(i,j) for j in I_list[:i] + I_list[i+1:]])

        constraints.extend([(len(label_list)-1,i) for i in I_list]) # last element is "O".
        model = BertCRF.from_pretrained(args.bert_model, PYTORCH_PRETRAINED_BERT_CACHE, num_labels=num_labels, constraints=constraints, include_start_end_transitions=False)
    else:
        model = BertForTokenClassification.from_pretrained(args.bert_model, PYTORCH_PRETRAINED_BERT_CACHE, num_labels=num_labels)

    if args.model_load != "":
        model.load_state_dict(torch.load(args.model_load))
        logger.info("Model state has been loaded.")

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)


    idtolabel = {}
    for (i, label) in enumerate(label_list):
        idtolabel[i] = label

    global_step = 0
    # best_mcc = 0.0
    best_acc = 0.0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                if args.use_crf:
                    crf_label_ids = label_ids.clone()
                    crf_label_ids[crf_label_ids == -1] = num_labels - 1 # Make -1's last label. This doesn't affect score, because these are masked.
                    loss, _ = model(input_ids, segment_ids, input_mask, crf_label_ids)
                else:
                    loss, _ = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

                # if global_step % args.val_each == 0:
                #     if args.do_eval:
                #         model.eval()
                #         # total_rates = np.array([0,0,0,0])
                #         all_preds = np.array([])
                #         all_label_ids = np.array([])
                #         eval_loss, eval_accuracy = 0, 0
                #         nb_eval_steps, nb_eval_examples = 0, 0
                #         for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                #             input_ids = input_ids.to(device)
                #             input_mask = input_mask.to(device)
                #             segment_ids = segment_ids.to(device)
                #             label_ids = label_ids.to(device)

                #             with torch.no_grad():
                #                 if args.use_crf:
                #                     crf_label_ids = label_ids.clone()
                #                     crf_label_ids[crf_label_ids == -1] = num_labels - 1 # Make -1's last label. This doesn't affect score, because these are masked.
                #                     tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, crf_label_ids)
                #                 else:
                #                     tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

                #             label_ids = label_ids.to('cpu').numpy()

                #             # logger.info(logits)

                #             eval_loss += tmp_eval_loss.mean().item()

                #             if not args.use_crf:
                #                 logits = logits.detach().cpu().numpy()
                #                 logits = np.argmax(logits, axis=-1).reshape(-1)

                #             label_ids = label_ids.reshape(-1)

                #             # logger.info(logits)
                #             if args.use_crf:
                #                 all_preds = np.append(all_preds, logits)
                #             else:
                #                 all_preds = np.append(all_preds, logits[label_ids != -1])

                #             all_label_ids = np.append(all_label_ids, label_ids[label_ids != -1])

                #             nb_eval_steps += 1

                #         eval_loss = eval_loss / nb_eval_steps
                #         eval_accuracy = accuracy(all_preds, all_label_ids)

                #         # precision, recall, f1, _ = precision_recall_fscore_support(all_label_ids, all_preds, average="macro", labels=list(range(0,num_labels)))
                #         precision, recall, f1 = evaluate([idtolabel[x] for x in all_label_ids.tolist()], [idtolabel[x] for x in all_preds.tolist()])
                #         mcc = matthews_corrcoef(all_preds, all_label_ids)
                #         result = {"eval_loss": eval_loss,
                #                   "eval_accuracy": eval_accuracy,
                #                   "precision": precision,
                #                   "recall": recall,
                #                   "f1": f1,
                #                   "mcc": mcc}

                #         # balanced_acc, f1_neg, f1_pos, mcc, _, _ = get_scores(total_rates.tolist())
                #         # result = {'eval_loss': eval_loss,
                #         #           'eval_accuracy': eval_accuracy,
                #         #           'global_step': global_step,
                #         #           'balanced_accuracy' : balanced_acc,
                #         #           'f1_neg' : f1_neg,
                #         #           'f1_pos' : f1_pos,
                #         #           'mcc' : mcc,
                #         #           'loss': tr_loss/nb_tr_steps}

                #         # if best_mcc < mcc:
                #         #     best_mcc = mcc
                #         if best_f1 < f1:
                #             best_f1 = f1
                #             logger.info("Saving model...")
                #             model_to_save = model.module if hasattr(model, 'module') else model  # To handle multi gpu
                #             torch.save(model_to_save.state_dict(), args.output_file)

                #         for key in sorted(result.keys()):
                #             logger.info("  %s = %.4f", key, result[key])

                #         model.train() # back to training


            if args.do_eval:
                model.eval()
                all_trigger_count = 0
                all_argument_count = 0
                corr_trigger_count = 0
                corr_argument_count = 0
                for input_ids, input_mask, segment_ids, doc_ids in eval_dataloader: # label_ids = None
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask)

                    # pdb.set_trace()
                    logits = logits.detach().cpu().numpy()
                    logits = np.argmax(logits, axis=-1)

                    for idx,doc_id in enumerate(doc_ids):
                        elem = dev_df.iloc[doc_id.item()]
                        if args.etype in elem.labels:
                            token_labels = logits[idx,:]
                            for event in elem.events:
                                if args.etype != event["event_type"]:
                                    continue
                                all_trigger_count += 1
                                curr_labels = token_labels[event["trigger"]["start"]:event["trigger"]["end"]]
                                if all([event["event_subtype"] in label_map[label] for label in curr_labels]):
                                    corr_trigger_count += 1

                                if args.task_name == "trigger":
                                    continue
                                for argument in event["arguments"]:
                                    all_argument_count += 1
                                    curr_labels = token_labels[argument["start"]:argument["end"]]
                                    if all([argument["role"] in label_map[label] for label in curr_labels]):
                                        corr_argument_count += 1


                if args.task_name == "trigger":
                    acc = corr_trigger_count / all_trigger_count
                else:
                    trigger_acc = corr_trigger_count / all_trigger_count
                    argument_acc = corr_argument_count / all_argument_count
                    acc = (trigger_acc + argument_acc) / 2

                if best_acc < acc:
                    best_acc = acc
                    logger.info("Saving model...")
                    model_to_save = model.module if hasattr(model, 'module') else model  # To handle multi gpu
                    torch.save(model_to_save.state_dict(), args.output_file)

                logger.info("***** Epoch " + str(epoch_num + 1) + " *****")
                logger.info("  accuracy = %.4f", acc)

                model.train() # back to training


    # for input_ids, input_mask, segment_ids, label_ids in train_dataloader:
    #     with open("sst-train.tsv", "a") as f:
    #         for i in range(len(input_ids)):
    #             f.write(str(input_ids[i].numpy().tolist()) + "\t" + str(input_mask[i].numpy().tolist()) + "\t" + str(segment_ids[i].numpy().tolist()) + "\t" + str(label_ids[i].numpy().tolist()) + "\n")

    # for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    #     with open("sst-dev.tsv", "a") as f:
    #         for i in range(len(input_ids)):
    #             f.write(str(input_ids[i].numpy().tolist()) + "\t" + str(input_mask[i].numpy().tolist()) + "\t" + str(segment_ids[i].numpy().tolist()) + "\t" + str(label_ids[i].numpy().tolist()) + "\n")

    if args.do_test:
        if args.use_crf:
            model = BertCRF.from_pretrained(args.bert_model, PYTORCH_PRETRAINED_BERT_CACHE, num_labels=num_labels, constraints=constraints, include_start_end_transitions=False)
        else:
            model = BertForTokenClassification.from_pretrained(args.bert_model, PYTORCH_PRETRAINED_BERT_CACHE, num_labels=num_labels)

        model.load_state_dict(torch.load(args.output_file))
        model.to(device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation on test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        model.eval()
        all_trigger_count = 0
        all_argument_count = 0
        corr_trigger_count = 0
        corr_argument_count = 0
        for input_ids, input_mask, segment_ids, doc_ids in test_dataloader:

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, axis=-1)

            for idx,doc_id in enumerate(doc_ids):
                elem = test_df.iloc[doc_id.item()]
                if args.etype in elem.labels:
                    token_labels = logits[idx,:]
                    for event in elem.events:
                        if args.etype != event["event_type"]:
                            continue
                        all_trigger_count += 1
                        curr_labels = token_labels[event["trigger"]["start"]:event["trigger"]["end"]]
                        if all([event["event_subtype"] in label_map[label] for label in curr_labels]):
                            corr_trigger_count += 1

                        if args.task_name == "trigger":
                            continue
                        for argument in event["arguments"]:
                            all_argument_count += 1
                            curr_labels = token_labels[argument["start"]:argument["end"]]
                            if all([argument["role"] in label_map[label] for label in curr_labels]):
                                corr_argument_count += 1

        if args.task_name == "trigger":
            acc = corr_trigger_count / all_trigger_count
        else:
            trigger_acc = corr_trigger_count / all_trigger_count
            argument_acc = corr_argument_count / all_argument_count

            acc = (trigger_acc + argument_acc) / 2

        logger.info("***** Test Eval results *****")
        logger.info("  accuracy = %.4f", acc)


if __name__ == "__main__":
    main()
