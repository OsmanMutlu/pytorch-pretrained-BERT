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

from pytorch_pretrained_bert.tokenization import printable_text, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(filename = '{}_log.txt'.format(datetime.datetime.now()),
                                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                        datefmt = '%m/%d/%Y %H:%M:%S',
                                        level = logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

runOutputFileName = "prediction.txt"

def main():
    """Main method of this module."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir")
    parser.add_argument("-o", "--output_file", default=None, type=str, required=True,
                        help="Output file for predictions")
    parser.add_argument("--bert_model", default="", type=str, required=True, help="Bert pre-trained model path")
    parser.add_argument("--bert_tokenizer", default="", type=str, required=True, help="Bert tokenizer path")
    parser.add_argument("--model_load",
                        default="",
                        type=str,
                        required=True,
                        help="The path of model state.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    args = parser.parse_args()

    max_seq_length = args.max_seq_length
    model_path = args.model_load

    input_file = args.input_file
    output_file = args.output_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    label_list = ["B-etime", "B-fname", "B-loc", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-loc", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O"]
    model = BertForTokenClassification.from_pretrained(args.bert_model, PYTORCH_PRETRAINED_BERT_CACHE, num_labels=len(label_list))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[i] = label

    # try:
    #     model.load_state_dict(torch.load(model_path)) # , map_location='cpu' for only cpu
    # except: #When model is parallel
    #     model = torch.nn.DataParallel(model)
    #     model.load_state_dict(torch.load(model_path)) # , map_location='cpu' for only cpu

    model.load_state_dict(torch.load(model_path))

    logger.info("Model state has been loaded.")

    model.to(device)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    examples = []
    words = []
    for (i, line) in enumerate(lines):
        line = line.strip()
        if line == "SAMPLE_START":
            words.append("[CLS]")
        elif line == "[SEP]":
            continue
        elif line == "":
            tokens = []
            for (j, word) in enumerate(words):
                if word == "[CLS]":
                    tokens.append("[CLS]")
                    continue

                tokenized = tokenizer.tokenize(word)
                tokens.append(tokenized[0])

            if len(tokens) > max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 1)]

            tokens.append("[SEP]")
            tokens = tokenizer.convert_tokens_to_ids(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            while len(tokens) < max_seq_length:
                tokens.append(0)
                segment_ids.append(0)
                input_mask.append(0)

            examples.append((tokens, input_mask, segment_ids))
            words = []
            continue
        elif line in ["\x91", "\x92", "\x97"]:
            continue
        else:
            words.append(line)

    # print(examples)

    all_labels = []
    model.eval()
    for (input_ids, input_mask, segment_ids) in examples:

        org_input_mask = input_mask
        org_input_mask = [x for x in org_input_mask if x != 0]

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            labels = np.argmax(logits, axis=-1).reshape(-1)

            labels = labels[0:len(org_input_mask)]
            # while len(labels) < max_seq_length:
            #     labels = np.append(labels, 16) # Add "O"

            all_labels = np.append(all_labels, labels)

    j = 0
    count = 0
    with open(output_file, "w", encoding="utf-8") as g:
        for (i, line) in enumerate(lines):
            line = line.strip()
            if line == "SAMPLE_START":
                count += 1
                g.write("SAMPLE_START\tO\n")
                j += 1
            elif line == "[SEP]":
                g.write("[SEP]\tO\n")
            elif line == "\x91":
                g.write("\x91\tO\n")
            elif line == "\x92":
                g.write("\x92\tO\n")
            elif line == "\x97":
                g.write("\x97\tO\n")
            elif line == "":
                g.write("\n")
                count = 0
                j += 1 # We have a SEP at the end
            else:
                count += 1
                if count < max_seq_length:
                    g.write(line + "\t" + label_map[all_labels[j]] + "\n")
                    j += 1
                else:
                    g.write(line + "\tO\n")

    logger.info("The predictions have been written to the output folder.")

if __name__ == '__main__':
    main()
