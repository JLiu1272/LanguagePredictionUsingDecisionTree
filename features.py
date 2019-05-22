"""
File: features.py
Goal: Extract
"""

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize, word_tokenize
import nltk
import string
import re
from enum import Enum
from nltk.corpus import brown

import csv

# 1 = English, 0 = Dutch
class Language(Enum):
    english = 1
    dutch = 0


# These are the most common english words
most_common_dutch = ['ik', 'je', 'dat', 'ze', 'hebben', 'weet', 'kan', 'ja', 'nee', 'bent', 'doen']

# Based on sources, these are the most common english nouns
most_common_english = ['area', 'book', 'business', 'case', 'child', 'company', 'country', 'day', 'eye', 'fact', 'family', 'government', 'group',
                       'hand', 'home', 'job', 'life', 'lot', 'man', 'money', 'month', 'mother', 'mr', 'night', 'number', 'part', 'people',
                       'place', 'point', 'problem', 'program', 'question', 'right', 'room', 'school', 'state', 'story', 'student', 'study',
                       'system', 'thing', 'time', 'water', 'way', 'week', 'woman', 'word', 'work', 'world', 'year']

dutch_vowel_combination = ["uu", "aa", "ieu", "ij", "ooi", "oei"]
english_vowel_combination = ["aw", "ay", "oy", "kn", "ph"]

dutch_suffix = ["ische", "thisch", "thie", "achtig", "aan", "iek", "ief", "ier", "iet", "een", "ant"]
eng_suffix = ["tion", "sion", "ial", "able", "ible", "ful", "acy", "ance", "ism", "ity", "ness", "ship", "ish", "ive", "less", "ious", "ify"]

dutch_prefix = []
eng_prefix = ["un"]


# These are the words that can appear before a possessive
# for dutch.
dutch_possessive = ['a', 'i', 'o', 'u', 's']

def contains_english_stopwords(sentence):
    """
    Checks if the sentence contains an english stop word.
    Return True if the sentence contains an english stop word
    and false otherwise
    :param sentence:
    :return:
    """
    stopwords_set = set(stopwords.words("english"))

    # Convert word into lower case and only consider
    # unique words
    words_set = set([word.lower() for word in sentence])

    common_elements = words_set.intersection(stopwords_set)

    if len(common_elements) > 0:
        return 1

    return 0


def contains_dutch_stopwords(sentence):
    """
    Checks if the sentence contains an dutch stop word.
    Return True if the sentence contains an dutch stop word
    and false otherwise
    :param sentence:
    :return:
    """
    stopwords_set = set(stopwords.words("dutch"))

    # Convert word into lower case and only consider
    # unique words
    words_set = set([word.lower() for word in sentence])

    common_elements = words_set.intersection(stopwords_set)

    if len(common_elements) > 0:
        return 0

    return 1

def vowel_combination_dutch(sentence):
    """
    Check to see if any words in sentence contain
    the ij pairing
    :param sentence:
    :param vowel:
    :return:
    """

    for word in sentence:
        for vowel in dutch_vowel_combination:
            if vowel in word:
                return 0

    # Nothing found, so return false
    return 1

def vowel_combination_eng(sentence):
    """
    Check to see if any words in sentence contain
    the ij pairing
    :param sentence:
    :param vowel:
    :return:
    """

    for word in sentence:
        for vowel in english_vowel_combination:
            if vowel in word:
                return 1

    # Nothing found, so return false
    return 0


def word_ending_dutch(sentence):
    """
    Check if any words in sentence ends with
    the ending provided by the dutch language

    :param sentence:
    :param end: The word that we want to end with
    :return:
    """

    for end in dutch_suffix:
        # When checking whether this exist or not,
        # we need to make sure that the word at least
        # has this length. Otherwise we skip it
        end_len = len(end)

        for word in sentence:
            if len(word) >= end_len and word[-end_len:] == end:
                return 0

    return 1

def word_ending_eng(sentence):
    """
    Check if any words in sentence ends with
    the ending provided by the english language

    :param sentence:
    :param end: The word that we want to end with
    :return:
    """

    # Check for the existence of english suffix
    for end in eng_suffix:
        end_len = len(end)

        for word in sentence:
            if len(word) >= end_len and word[-end_len:] == end:
                #print("Found English ending:")
                #print(word)
                #rint(word[-end_len:])
                #print()
                return 1

    return 0

def clean_sentence(sentence):
    """
    Tokenize the sentence, and remove punctuation.
    Return a list of words in sentence.

    :param sentence:
    :return:
    """
    # Strip punctuations
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # Tokenize sentence
    tokens = wordpunct_tokenize(sentence)

    return tokens

def contains_common_dutch(sentence):
    """
    Determines whether the sentence
    contains that word. If it contains the word from the dutch list,
    return that it is a dutch word.
    :param sentence:
    :return:
    """

    # if the sentence contains a common dutch word
    # return dutch else english
    # If word does not exist, check
    for word in sentence:
        if word in most_common_dutch:
            return 0

    return 1

def contains_common_eng(sentence):
    """
    Determines whether the sentence
    contains that word. If it contains the word from the dutch list,
    return that it is a dutch word.
    :param sentence:
    :return:
    """

    # if the sentence contains a common dutch word
    # return dutch else english
    # If word does not exist, check
    for word in sentence:
        if word in most_common_english:
            return 1

    return 0

def common_letter_eng(sentence):

    num_e = 0

    for word in sentence:
        if word in 'y':
            num_e += 1

    if num_e >= 10:
        return 1

    return 0

def contains_prefix_eng(sentence):
    for word in sentence:
        if word[:-2] == 'un':
            return 1

    return 0

def get_idx(sentence, word):
    """
    Get the index in which the word is
    found in the sentence
    If no word found, return -1
    :param sentence:
    :param word:
    :return:
    """

    for i, w in enumerate(sentence):
        if w == word:
            return sentence[i-1][-1], i

    return -1

def possessive_prounouns_eng(sentence):
    """
    Does it contain a possessive pronoun. If it does
    is the previous word a vowel or an s. If it is,
    it is dutch, otherwise it is English
    :param sentence:
    :return:
    """
    # The sentence contains a possessive pronouns
    has_possessive = get_idx(sentence, "'s")

    if has_possessive != -1:
        return 1

    return 0

def most_common_words(text, lang):
    """
    Extract the top 20 most common words in the given text
    :param text:
    :return:
    """

    f = open(text, "r")
    allWords = []
    stopwords_set = stopwords.words(lang)

    for line in f.readlines():
        # Strip letters that are lower
        line = line.lower()

        # String punctuation
        line = line.translate(str.maketrans('', '', string.punctuation))

        # Strip away quations
        for stripper in ["\"", "\'", "\"", '\'', '\”', '„', '’']:
            line = re.sub(stripper, '', line)

        tokens = word_tokenize(line)
        allWords += tokens

    allWordDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords_set)
    mostCommon = allWordDist.most_common(15)
    print([tup[0] for tup in mostCommon])

def most_common_noun(text, lang):
    """
    Grab the 50 most common nouns in the particular language
    :param text:
    :param lang:
    :return:
    """
    f = open(text, "r")
    allNouns = []
    stopwords_set = stopwords.words(lang)


    for line in f.readlines():
        # Convert word into lower letters
        line = line.lower()

        # If the word is already in stop word,
        # do not include it
        if line in stopwords_set:
            continue

        # Remove spaces and append to list
        allNouns.append(line.strip())

    return allNouns

def write_to_csv(des, src, test=False):
    """
    Write the result based on all the different attributes into a CSV
    :param file:
    :return:
    """

    title = ["CommonDutch", "CommonEng",
             "VowelCombDutch", "VowelCombEng",
             "StopwordsDutch", "StopwordsEng",
             "EndDutch", "EndEng",
             "Lang"]


    # Result for one sentence for each of the test cases
    row_res = []

    # Write title
    # Write data to csv
    with open(des, "w") as data:
        writer = csv.writer(data)
        writer.writerow(title)

    # The training data
    raw = open(src, "r")

    for line in raw.readlines():

        # Only do this if it is not a test file
        # for training process
        if test is False:
            tokens = line.split("|")
            lang = tokens[0].strip()
            sent = tokens[1].strip()
        else:
            sent = line

        token_sent = clean_sentence(sent)
        row_res.append(contains_common_dutch(token_sent))
        row_res.append(contains_common_eng(token_sent))
        row_res.append(vowel_combination_dutch(token_sent))
        row_res.append(vowel_combination_eng(token_sent))
        row_res.append(contains_dutch_stopwords(token_sent))
        row_res.append(contains_english_stopwords(token_sent))
        row_res.append(word_ending_dutch(token_sent))
        row_res.append(word_ending_eng(token_sent))

        if test is False:
            if lang == "en":
                row_res.append(1)
            else:
                row_res.append(0)

        # Write data to csv
        with open(des, "a") as data:
            writer = csv.writer(data)
            writer.writerow(row_res)

        # Result row
        row_res = []


def test_function(raw_file):
    # The training data
    raw = open(raw_file, "r")

    for line in raw.readlines():
        tokens = line.split("|")
        lang = tokens[0].strip()
        sent = tokens[1].strip()
        print(lang)
        print(sent)
        print()

        token_sent = clean_sentence(sent)
        #row_res.append(contains_common_dutch(token_sent))
        #row_res.append(contains_common_eng(token_sent))
        #row_res.append(vowel_combination_dutch(token_sent))
        #row_res.append(vowel_combination_eng(token_sent))
        #row_res.append(contains_dutch_stopwords(token_sent))
        #row_res.append(contains_english_stopwords(token_sent))
        #row_res.append(word_ending_dutch(token_sent))
        print(word_ending_eng(token_sent))


#print(most_common_words("english_raw2.txt", "english"))

sentence_eng = "to himself , dwelling especially upon the character and actions of that strange being who had played the rôle of monarch . Harry 's light and"
sentence_dutch = "Clasie werd een kwartier voor tijd gewisseld omdat Van Bronckhorst met Van Persie als extra aanvaller op overwinning wilde jagen. Dat accepteerde de huurling van" \
                 "Southampton aanvankelijk nieteen."

#most_common_words("raw_data/dutch/dutch_raw1.txt", "dutch")
#most_common_english = most_common_noun("raw_data/english/english_common_nouns.txt", "english")
#print(most_common_english)

tokens_eng = clean_sentence(sentence_eng)
tokens_dutch = clean_sentence(sentence_dutch)


# Testing suffixes
#print(word_ending_dutch(tokens_dutch))
#print(word_ending_eng(tokens_dutch))

# Testing vowel combinations
#print(vowel_combination_eng(tokens_eng))
#print(vowel_combination_dutch(tokens_dutch))

# Testing for stop words
#print(contains_dutch_stopwords(tokens_dutch))
#print(contains_english_stopwords(tokens_eng))

# Testing for common words
#print(contains_common_dutch(tokens_dutch))
#print(contains_common_eng(tokens_eng))


"""
print("Dutch")
print(possessive_prounouns_dutch(tokens_dutch))
print(word_length_dutch(tokens_dutch))
print(contains_common_dutch(tokens_dutch))
print(vowel_combination_dutch(tokens_dutch))
print(contains_dutch_stopwords(tokens_dutch))
print(word_ending_dutch(tokens_dutch))

print("English")
print(word_length_eng(tokens_eng))
print(contains_common_eng(tokens_eng))
print(vowel_combination_eng(tokens_eng))
print(contains_english_stopwords(tokens_eng))
print(word_length_eng(tokens_eng))
"""

write_to_csv("processed_data/binary_dutch_eng.csv", "processed_data/train.txt")
write_to_csv("processed_data/binary_dutch_eng_validation.csv", "processed_data/test.txt")

#test_function("processed_data/train.txt")

#print(Language.dutch)