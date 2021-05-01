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

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.translate(str.maketrans("","",string.punctuation))
        if new_word != '':
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

def read_data(path):
    contents = []
    for filename in os.listdir(path):
        data = strip_html(open(path+'/'+filename,'r', errors='ignore').read())
        contents.append((filename,data))
    return contents


def get_vocabulary(data):
    tokens = []
    with open(os.path.join(os.getcwd(),"vocabulary1.txt"),"r") as rf:
        tokens = rf.read().split()
    return tokens

def preprocess_data(contents):
    dataDict = {}
    for content in contents:
        sample = content[1]
        sample = sample.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        tokens = word_tokenize(sample)
        lexical = lexical_analysis(tokens)
        filtered_tokens = remove_stopwords(lexical)
        stemmed_tokens = stem_words(filtered_tokens)
        filtered_tokens1 = remove_stopwords(stemmed_tokens)
        dataDict[content[0]] = filtered_tokens1
    return dataDict

def generate_inverted_index(data):
    all_words = get_vocabulary(data)
    index = {}
    for word in all_words:
        index[word] = {}
        for doc, tokens in data.items():
            index[word][doc] = tokens.count(word)
    return index

data = read_data("Docs")
preprocessed_data = preprocess_data(data)
inverted_index = generate_inverted_index(preprocessed_data)

inverted_index

inverted_index_df = pd.DataFrame(inverted_index).T

print(inverted_index_df)

inverted_index_df.to_excel("inverted_index.xlsx")

while(True):
    words = input("Enter words for inverted index : ").split()
    words = lexical_analysis(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = remove_stopwords(words)

    for word in words:
        print("\n")
        if word in inverted_index.keys():
            print(word,inverted_index[word])
        else:
            print(word,"Not Found")
    print("\n")
    choice = int(input("Enter 1 for continue...."))
    if(choice!=1):
        break

