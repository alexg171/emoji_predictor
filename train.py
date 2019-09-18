# script that trains model
# ex run: python train.py datasets/1k_tweets.txt datasets/1k_labels.txt

# standard library
import sys
import time
# numpy
import numpy as np
# scikit imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

import test

## reads data and generates a machine learning classifier
# @param data    file containing different tweet on every line
# @param labels  file containing differen emoji on every line, corresponds to tweet
def train(data,labels):
    print("Training...")
    # read in data
    tweets = open(data, encoding="utf8").read().split('\n')
    emojis = open(labels, encoding="utf8").read().split('\n')

    # learn the vocabulary dictionary and return term-document matrix
    count_vect = CountVectorizer()
    term_doc_matrix = count_vect.fit_transform(tweets)

    # normalize term document matrix
    tfidf_transformer = TfidfTransformer()
    normalized_tdm = tfidf_transformer.fit_transform(term_doc_matrix)

    # timer
    start_t = time.time()
    
    # classifiers
    # clf = MultinomialNB().fit(normalized_tdm, emojis)
    # clf = SGDClassifier().fit(normalized_tdm, emojis)
    # clf = svm.SVC().fit(normalized_tdm, emojis)
    clf = LogisticRegression().fit(normalized_tdm, emojis)
    # clf = KNeighborsClassifier().fit(normalized_tdm, emojis)
    # clf = DecisionTreeClassifier().fit(normalized_tdm, emojis)
    # clf = MLPClassifier().fit(normalized_tdm, emojis)

    # store model
    joblib.dump(clf, 'out/test_classifier.joblib')
    
    # end timer to measure how long it took to train
    end_t = time.time()
    trainingtime = end_t - start_t
    print("Training time: %s" % trainingtime)

    return 'out/test_classifier.joblib'

def main():
    if len(sys.argv) < 3:
        print("Wrong number of arguments")
        exit()

    data = sys.argv[1]
    labels = sys.argv[2]
    
    train(data,labels)

    test.test(data, labels,'out/test_classifier.joblib' )


if __name__ == "__main__":
    main()
