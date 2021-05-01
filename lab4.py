
import pandas as pd
import nltk
import itertools
import math
import operator
from statistics import mean
from nltk.corpus import stopwords
from nltk.stem import *
import os,sys
import re, string, unicodedata
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    new_words = []
    for word in words:
        new_word = re.sub(r'\d+','',word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    stop_words = set(stopwords.words("english"))
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lexical_analysis(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_numbers(words)
    return words

def preprocess_query(query):
    sample = query
    sample = sample.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    tokens = word_tokenize(sample)
    lexical = lexical_analysis(tokens)
    filtered_tokens = remove_stopwords(lexical)
    stemmed_tokens = stem_words(filtered_tokens)
    filtered_tokens1 = remove_stopwords(stemmed_tokens)
    return filtered_tokens1


df = pd.read_excel("inverted_index.xlsx",index_col="Unnamed: 0")
inverted_index = df.copy()

print(inverted_index)

N = df.shape[1]
nw = df.astype(bool).sum(axis=1)
prob = (N-nw+0.5)/(nw+0.5)


query = input("Enter the query : ")
query = strip_html(query)
query_words = preprocess_query(query)
qw = set(query_words)

similarity = {}
similarity["query"] = {}
for column in df:
    ll = (df[df[column]>0].index).intersection(qw)
    similarity["query"][column]=prob[ll].product()

result = pd.DataFrame(similarity).sort_values("query",ascending=False)

# For confirming whether above ranking is correct lets look at inverted index of query terms
for word in qw:
    if word in inverted_index.index:
        print("\n{} Index: \n".format(word))
        print("{} \n".format(inverted_index.loc[word]))
    else:
        print(word,"No entry in inverted index")


for word in qw:
    print("Probabilities of {} : {}".format(word,prob[word]))

