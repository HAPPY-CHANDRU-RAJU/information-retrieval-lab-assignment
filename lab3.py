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

# Preprocessing query

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

print(df)

def normalize_tf(df):
    for column in df:
        m = max(df[column])
        if m!=0:
            df[column] = df[column]/m
    return df

def calculate_idf(df):
    idf_score = {}
    N = df.shape[1]
    all_words = df.index
    word_count = df.astype(bool).sum(axis=1)
    for word in all_words:
        idf_score[word] = math.log10(N/word_count[word])
    return idf_score

def calculate_tfidf(data, idf_score):
    scores = {}
    for key,value in data.items():
        scores[key] = data[key]
    for doc,tf_scores in scores.items():
        for token, score in tf_scores.items():
            tf = score
            idf = idf_score[token]
            tf_scores[token] = tf * idf
    return scores


normalized_tf = normalize_tf(df)
idf_score = calculate_idf(normalized_tf)
tf_idf_docs = calculate_tfidf(normalized_tf,idf_score)


query = input("Enter the query : ")
query = strip_html(query)
query_words = preprocess_query(query)


def tf_query(query_words):
    all_words = df.index
    index = {}
    index["query"] = {}
    for word in all_words:
        index["query"][word] = 0
    for qword in query_words:
        if qword in all_words:
            index["query"][qword] = query_words.count(qword)
    return index


tf_for_query = tf_query(query_words)
tf_for_query = pd.DataFrame(tf_for_query)
normalized_tf_for_query = normalize_tf(tf_for_query)
tf_idf_query = calculate_tfidf(normalized_tf_for_query,idf_score)
tf_idf_query = pd.DataFrame(tf_idf_query)
tf_idf_docs = pd.DataFrame(tf_idf_docs)


print(tf_idf_docs)

print(tf_idf_query)

def get_similarity(tf_idf_docs,tf_idf_query):
    query_docs = {}
    query_docs["query"] = {}
    query_length = math.sqrt(sum(tf_idf_query.loc[value] ** 2 for value in tf_idf_query.index))
    
    if(query_length==0):
        print("Your terms in query did not match any document")
        for column in tf_idf_docs:
            query_docs["query"][column] = 0
        return query_docs
    
    for column in tf_idf_docs:
        
        num = 0
        sum_of_squares = 0
        
        for value in tf_idf_docs.index :
            sum_of_squares+=tf_idf_docs[column].loc[value] ** 2
            num+= tf_idf_docs[column].loc[value] * tf_idf_query["query"].loc[value] 
            
        doc_len = math.sqrt(sum_of_squares)
        cosine_sim = num/(doc_len*query_length)
        query_docs["query"][column] = cosine_sim
    
    return query_docs

rank = get_similarity(tf_idf_docs,tf_idf_query)
rank = pd.DataFrame(rank).sort_values("query",ascending=False)

print(rank)
# For confirming whether above ranking is correct lets look at inverted index of query terms
for word in query_words:
    print("\n")
    if word in inverted_index.index:
        print(word)
        print(inverted_index.loc[word])
    else:
        print(word,"No entry in inverted index")
    print("\n")

def compare_documents(tf_idf_docs):
    compare = {}
    for column1 in tf_idf_docs:
        compare[column1] = {}
        query_length = math.sqrt(sum(tf_idf_docs[column1].loc[value] ** 2 for value in tf_idf_docs.index))
    
        for column in tf_idf_docs:
        
            num = 0
            sum_of_squares = 0
        
            for value in tf_idf_docs.index :
                sum_of_squares+=tf_idf_docs[column].loc[value] ** 2
                num+= tf_idf_docs[column].loc[value] * tf_idf_docs[column1].loc[value] 
            
            doc_len = math.sqrt(sum_of_squares)
            cosine_sim = num/(doc_len*query_length)
            compare[column1][column] = cosine_sim
    
    return compare

compare = compare_documents(tf_idf_docs)
compare = pd.DataFrame(compare)
print(compare)
compare.to_excel("comparison_among_documents.xlsx")
