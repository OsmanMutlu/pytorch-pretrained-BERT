#!/usr/bin/env python

"""Random baseline for the PAN19 hyperpartisan news detection task"""
# Version: 2018-09-24

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputDir=<directory>
#   Directory to which the predictions will be written. Will be created if it does not exist.

from __future__ import division

import os
import sys
from lxml import etree
import codecs
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import argparse
import logging
import datetime
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert.tokenization import printable_text, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(filename = '{}_log.txt'.format(datetime.datetime.now()),
                                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                        datefmt = '%m/%d/%Y %H:%M:%S',
                                        level = logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

runOutputFileName = "prediction.txt"

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_test_examples(self, inputFile):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        lines = pd.read_csv(input_file)
        return lines

class HyperProcessor(DataProcessor):
    """Processor for the Hyperpartisan data set."""

    def get_test_examples(self, inputFile):
        """See base class."""
        return self._create_examples(
            self._read_tsv(inputFile), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in lines.iterrows():
            guid = i
            text_a = line.text
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        return examples

class EmwProcessor(DataProcessor):
    """Processor for the Emw data set."""

    def get_test_examples(self, inputFile):
        """See base class."""
        return self._create_examples(
            self._read_tsv(inputFile), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in lines.iterrows():
            # guid = i
            guid = line.item_id
            text_a = str(line.text)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        return examples


def convert_examples_to_features(example, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

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
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
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

class HyperpartisanData(Dataset):
    """"""
    def __init__(self, examples, max_seq_length, tokenizer):
        self.examples = examples
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        feats = convert_examples_to_features(ex, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor(feats.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feats.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feats.segment_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, ex.guid

def main():
    """Main method of this module."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--inputFile",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir")
    parser.add_argument("-o", "--outputFile", default=None, type=str,
                        help="Output file for predictions")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="emw",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--model_load",
                        default="",
                        type=str,
                        required=True,
                        help="The path of model state.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Batch size.")

    args = parser.parse_args()

    processors = {
        "hyperpartisan": HyperProcessor,
        "emw": EmwProcessor,
    }

    bert_model = args.bert_model
    max_seq_length = args.max_seq_length
    model_path = args.model_load
    batch_size = args.batch_size
    task_name = args.task_name.lower()
    processor = processors[task_name]()

    inputFile = args.inputFile
    outputFile = args.outputFile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForSequenceClassification.from_pretrained(bert_model, PYTORCH_PRETRAINED_BERT_CACHE)
    try:
        model.load_state_dict(torch.load(model_path)) # , map_location='cpu' for only cpu
    except: #When model is parallel
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path)) # , map_location='cpu' for only cpu

    logger.info("Model state has been loaded.")

    model.to(device)

    test_examples = processor.get_test_examples(inputFile)
    random.shuffle(test_examples)

    test_dataloader = DataLoader(dataset=HyperpartisanData(test_examples, max_seq_length, tokenizer), batch_size=batch_size)

    df = pd.DataFrame(columns=["item_id","Bert_prediction"])
    # df = pd.DataFrame(columns=["id","prediction"])

    model.eval()
    for input_ids, input_mask, segment_ids, doc_ids in test_dataloader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            labels = np.argmax(logits, axis=1)

        df = df.append([{"item_id":doc_ids[i], "Bert_prediction":labels[i]} for i in range(len(labels))], ignore_index=True)
        # df = df.append([{"id":doc_ids[i], "prediction":labels[i]} for i in range(len(labels))], ignore_index=True)

    inp = pd.read_csv(inputFile)
    inp.item_id = inp.item_id.astype('int64')
    df.item_id = df.item_id.astype('int64')
    df = inp.merge(df, on="item_id")
    # df.to_csv(outputFile, index=False, header=None)
    df.to_csv(outputFile, index=False)
    logger.info("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main()
