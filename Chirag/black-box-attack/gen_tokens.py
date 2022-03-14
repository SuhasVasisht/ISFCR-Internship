from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import os
import argparse

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from gensim.models import Word2Vec

import numpy as np
import pandas as pd


def preprocess(spam_data_path, output_word2vec_path):

    stop_words = stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    df = pd.read_csv(spam_data_path, encoding='latin-1')

    df['patterns'] = df['Input'].apply(
        lambda x: ' '.join(x.lower() for x in x.split()))
    df['patterns'] = df['patterns'].apply(lambda x: ' '.join(
        x for x in x.split() if x not in string.punctuation))
    df['patterns'] = df['patterns'].str.replace('[^\w\s]', '')
    df['patterns'] = df['patterns'].apply(
        lambda x: ' '.join(x for x in x.split() if not x.isdigit()))
    df['patterns'] = df['patterns'].apply(lambda x: ' '.join(
        x for x in x.split() if not x in stop_words))
    df['patterns'] = df['patterns'].apply(lambda x: " ".join(
        [wordnet_lemmatizer.lemmatize(word) for word in x.split()]))
    df['patterns'] = df.apply(
        lambda row: nltk.word_tokenize(row['patterns']), axis=1)

    size = 1000
    window = 3
    min_count = 1
    workers = 3
    sg = 1

    start_time = time.time()
    tokens = pd.Series(df['patterns']).values

    # Train the Word2Vec Model
    w2v_model = Word2Vec(tokens, min_count=min_count, size=size,
                         workers=workers, window=window, sg=sg)
    print("Time taken to train word2vec model: " + str(time.time() - start_time))

    word2vec_model_file = 'word2vec_' + str(size) + '.model'
    w2v_model.save(word2vec_model_file)

    word2vec_model_file = 'word2vec_' + str(size) + '.model'

    # Load the model from the model file
    sg_w2v_model = Word2Vec.load(word2vec_model_file)

    print("Total number of words is")
    print(len(sg_w2v_model.wv.vocab))

    word2vec_filename = 'dataset/all_review_word2vec.csv'

    with open(word2vec_filename, 'w+') as word2vec_file:
        for index, row in df.iterrows():
            model_vector = (np.mean([sg_w2v_model[token]
                            for token in row['patterns']], axis=0)).tolist()

            if index == 0:
                header = ",".join(str(ele) for ele in range(1000))
                word2vec_file.write(header)
                word2vec_file.write("\n")

            if type(model_vector) is list:
                line1 = ",".join([str(vector_element)
                                  for vector_element in model_vector])
            else:
                line1 = ",".join([str(0) for i in range(size)])
            word2vec_file.write(line1)
            word2vec_file.write('\n')

    word2vec_df = pd.read_csv(word2vec_filename)
    word2vec_df['Result'] = df['Result']

    # Encoding labels
    word2vec_df = pd.get_dummies(word2vec_df, columns=['Result'])
    print(word2vec_df.head(10))

    word2vec_df.to_csv(output_word2vec_path)
    print("Saved Word2Vec sequences")
    os.remove(word2vec_filename)


if __name__ == '__main__':
    # dataset/spam_synth_data.csv

    parser = argparse.ArgumentParser(
        description="""python gen_tokens [--input | -i] (str) [--output | -o] (str)""")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the input spam dataset.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to the output, where the word2vec sequences are stored.')

    args = parser.parse_args()
    in_path = args.input
    out_path = args.output

    preprocess(in_path, out_path)
