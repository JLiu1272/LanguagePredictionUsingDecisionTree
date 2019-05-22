"""
File: data_collection.py
Objective: Grab English and Dutch sentences
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import requests
import time
import sys
import string
import nltk

NUM_WORD_LIMIT = 15

def grab_data(raw, train, test, en):

    raw = open(raw, "r")

    sentences = []
    sentence = []

    for line in raw:
        if line != "\n":
            # Remove numbers
            line = re.sub('\d+', '', line)

            # Remove end lines
            line = re.sub("\n", "", line)

            # Remove brackets
            line = re.sub("\[", "", line)
            line = re.sub("\]", "", line)
            line = re.sub("\|", "", line)

            # Punctuation
            #line = line.translate(None, string.punctuation)

            # Tokenize
            tokenize = nltk.word_tokenize(line)

            if len(sentence) >= NUM_WORD_LIMIT:
                data = en + " |" + " ".join(sentence)
                sentences.append(data)
                sentence = []
            else:
                sentence += tokenize

    split_idx = int(len(sentences)*0.8)

    with open(train, 'a') as f:
        for sentence in sentences[:split_idx]:
            f.write("%s\n" % sentence)

    with open(test, 'a') as f:
        for sentence in sentences[split_idx:]:
            f.write("%s\n" % sentence)


    print("Total train sentences:")
    print(len(sentences[:split_idx]))

    print("Total test sentences:")
    print(len(sentences[split_idx:]))

def main():
    grab_data("raw_data/dutch/dutch_raw1.txt", "train.txt", "test.txt", "nl")
    grab_data("raw_data/english/english_raw2.txt", "train.txt", "test.txt", "en")
    grab_data("raw_data/dutch/dutch_news_raw4.txt", "train.txt", "test.txt","nl")
    grab_data("raw_data/english/english_news_raw3.txt", "train.txt", "test.txt", "en")

main()