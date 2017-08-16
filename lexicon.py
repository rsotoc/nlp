# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:26:29 2016

@author: rsotoc
"""

from collections import OrderedDict
import operator
import re
import math
import copy
import argparse
from pathlib import Path

import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from bs4 import BeautifulSoup

#-----------------------------------------------------------------------
working_df = pd.DataFrame(columns=["title", "description"])

TECHNICAL = ['external links', 'jpg thumb', 'align center', 'align right',
         'thumb right', 'right align', 'center align', 'thumb px', 'align left', 
         'class wikitable', 'scope row', 'style background', 'references external',
         'right px', 'right thumb', 'thumb upright', 'thumb left', 'links official',
         'wikitable sortable', 'px right', 'text align', 'urlappend bseq', 
         'scope col', 'class unsortable', 'center rowspan', 'jpg thumbnail']

APOSTROFOS = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
              "didn't": "did not", "doesn't": "does not", "don't": "do not",
              "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
              "he's": "he is", "I'll": "I will", "I'm": "I am",
              "I've": "I have", "isn't": "is not", "it's": "it is",
              "let's": "let us", "mustn't": "must not", "shan't": "shall not",
              "she'll": "she will", "she's": "she is",
              "shouldn't": "should not", "that's": "that is",
              "there's": "there is", "they're": "they are",
              "they've": "they have", "we're": "we are", "we've": "we have",
              "weren't": "were not", "what're": "what are",
              "what's": "what is", "what've": "what have",
              "where's": "where is", "who're": "who are",
              "who's": "who is", "who've": "who have", "won't": "will not",
              "wouldn't": "would not", "you're": "you are",
              "you've": "you have"}


#-----------------------------------------------------------------------
def xml_to_dataframe(origen):
    tree = ET.parse(origen)

    root = tree.getroot()
    index = 0
    for child in root:
        if child.tag.find("page") >= 0:
            for grandchild in child:
                if grandchild.tag.find("title") >= 0:
                    title = grandchild.text
                if grandchild.tag.find("revision") >= 0:
                    for grand2child in grandchild:
                        if grand2child.tag.find("text") >= 0:
                            text = grand2child.text.lower()
                            #Eliminar las cadenas que inician en {{ seguidas de
                            #cualquier cosa excepto }} y terminadas con }}
                            text = re.sub('{{[^}}]*}}', '', text)
                            #Misma idea, pero con el caracter especial \[ \] y Category:
                            text = re.sub('\[\[Category:[^\]\]]*\]\]', '', text)
                            #... y entre === ===
                            text = re.sub('={3}[\w]+={3}', '', text)
                            #... y entre == ==
                            text = re.sub('={2}[\w]+={2}', '', text)
                            text = BeautifulSoup(text, "lxml").get_text()
                            #Eliminar direcciones http
                            text = re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
                            #... y direcciones wikt*
                            text = re.sub('\[\[wikt[^|]*|', '', text)
                            #... y direcciones de correo
                            text = re.sub('[\w\.-]+@[\w\.-]+', " ", text)
                            #Eliminar puntos decorativos como en S.H.I.E.L.D.
                            text = text.replace(".", "")

                            words = text.split()
                            texto = [APOSTROFOS[word]
                                     if word in APOSTROFOS else word for word in words]
                            texto = " ".join(texto)
                            texto = re.sub("[^a-zA-Z]", " ", texto)
                            #Eliminar palabras repetidas consecutivas
                            words = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', texto).split()
                            texto = " ".join(words)

            working_df.loc[index] = [title, texto]
            index = index + 1

    stops = set(stopwords.words("english"))
    working_df["words_wsw"] = list(map(lambda row:
        [w for w in row.split() if not w in stops and len(w) > 0],
         working_df.description))

    return

#-----------------------------------------------------------------------
def count_tokens_in_dfcol(column):
    all_words = []
    for row in column:
        all_words.extend(row)

    return nltk.FreqDist(all_words)

#-----------------------------------------------------------------------
def describe_corpus(column):
    counter = count_tokens_in_dfcol(column)
    print("Cantidad de tokens en el corpus: ", counter.N())
    print("Cantidad de tokens diferentes: ", len(counter.most_common()))
    print("tokens utilizados una vez: ", counter.Nr(1))
    print("\nTokens más populares:\n", counter.most_common(50))

    return counter

#-----------------------------------------------------------------------
def get_idf_dict(tokens_list, column):
    idf_dict = dict(zip(tokens_list, [0]*len(tokens_list)))
    for w in tokens_list:
        for d in column:
            if w in d:
                idf_dict[w] = idf_dict.get(w) + 1

    return idf_dict

#-----------------------------------------------------------------------
def get_useful_tokens(counter, column):
    global working_df
    
    useless_words = counter.hapaxes()
    all_tokens = [w[0] for w in counter.most_common() if not w[0] in useless_words]

    idf_dict = get_idf_dict(all_tokens, column)

    num_docs = len(column)
    for w in idf_dict.keys():
        v = num_docs / idf_dict.get(w)
        idf_dict[w] = math.log(v, 2)

    idf_dict2 = copy.copy(idf_dict)
    for w in idf_dict2.keys():
        idf_dict2[w] = idf_dict2.get(w) * counter.freq(w)

    idf_dict2 = OrderedDict(sorted(idf_dict2.items(),
                                   key=operator.itemgetter(1), reverse=True))
    keys = list(idf_dict2.keys())
    values = list(idf_dict2.values())
    new_keys = [item[0] for item in zip(keys, values)
                if not ((item[1] < 0.001 and len(item[0]) <= 2) or item[1] < 5e-5)]

    working_df["main_words"] = list(map(lambda row:
        [w for w in row if w in new_keys], working_df.words_wsw))
    working_df = working_df.drop('words_wsw', axis=1)


#-----------------------------------------------------------------------
def get_top_collocations():
    global working_df

    working_df["clean_bigrams"] = list(map(lambda words, desc:
        [b for b in list(ngrams(words, 2))
        if b in list(ngrams(word_tokenize(desc), 2))],
        working_df.main_words, working_df.description))
        
    working_df = working_df.reindex(columns=
                         ["title", "description", "main_words", "clean_bigrams",
                         "all_collocations", "new_description"])
        
    working_df['all_collocations'] = working_df['all_collocations'].astype(list)
    for index, row in zip(range(len(working_df)), 
                          zip(working_df.clean_bigrams, working_df.description)):
        collocations = []
        for b in row[0]:
            s = b[0]+" "+b[1]
            if (not s in TECHNICAL) and s in row[1]:
                collocations.append(s)
        working_df.set_value(index, 'all_collocations', collocations)
        
    counter = count_tokens_in_dfcol(working_df.all_collocations)
    collocations = list(counter.keys())
    for w in reversed(collocations):
        if len(w) < 5 or counter.freq(w) * counter.N() < 10:
            collocations.remove(w)
            
    idf_dict = get_idf_dict(collocations, working_df.all_collocations)
    top_collocations = []
    num_docs = len(working_df)
    for d in working_df.description:
        N = len(d.split()) - 1 #El número de bigramas es le núymero de palabras - 1
        for w in reversed(collocations): #En reversa para evitar problemas con los índices
            if w in d:
                tfidf = d.count(w) / N * math.log(num_docs/idf_dict[w], 2)
                if tfidf > 0.01:
                    top_collocations.append(w)
                    collocations.remove(w)

    for i, row in zip(range(num_docs), working_df.main_words):
        s = " ".join(row)
        for w in top_collocations:
            s = re.sub(" " + w, " " + "_".join(w.split()), s)
        working_df.loc[i, "new_description"] = s


#-----------------------------------------------------------------------
def main(origen, destino):
    if not Path(origen).exists():
        print("Nop")
        return

    xml_to_dataframe(origen)

#Step 1
    tokens_counter = describe_corpus(working_df.words_wsw)
    get_useful_tokens(tokens_counter, working_df.words_wsw)
    get_top_collocations()
    describe_corpus(working_df.all_collocations)
#Step 2
    stops = set(stopwords.words("english"))
    working_df["words_wsw"] = list(map(lambda row: 
        [w for w in row.split() if not w in stops and len(w) > 0], 
         working_df.new_description))
    tokens_counter = count_tokens_in_dfcol(working_df.words_wsw)
    get_useful_tokens(tokens_counter, working_df.words_wsw)
    describe_corpus(working_df.main_words)
    
    #Guardar La base de datos para posteriores usos
    working_df.to_json(destino, orient='records')

#-----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="archivo XML de entrada")
    parser.add_argument("-o", "--output", help="archivo JSON de salida")
    args = parser.parse_args()

    main(args.input, args.output)


