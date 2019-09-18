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

## runs a k-fold validation on classifier (?) idk if i did it right XD
# @param data     file containing different tweet on every line
# @param labels   file containing differen emoji on every line, corresponds to tweet
# @param clffile  path to classifier model output file
def test(data, labels, clffile):
    print("Testing...")

    # load data
    tweets = open(data, encoding="utf8").read().split('\n')
    emojis = open(labels, encoding="utf8").read().split('\n')

    # fit data
    count_vect = CountVectorizer()
    tfidf_transform = TfidfTransformer()
    tdm = count_vect.fit_transform(tweets)
    tfidf_transform.fit_transform(tdm)

    # load classifier
    clf = joblib.load(clffile)

    # k-fold cross validation
    for i in range(10):
        start = i*100
        end = start + 100

        print ("***** Iteration: %s *****" % i)
        print("Range: ", start, end)

        docs_test = tweets[start:end]
        emojis_test = emojis[start:end]

        tdm = count_vect.transform(docs_test)
        test_normalized_tdm = tfidf_transform.transform(tdm)
        
        predicted = clf.predict(test_normalized_tdm)

        print("Accuracy: ", np.mean(predicted == emojis_test))

def main():
    if len(sys.argv) < 3:
        print("Wrong number of arguments")
        exit()

    data = sys.argv[1]
    labels = sys.argv[2]
    
    test(data,labels, 'out/test_classifier.joblib')

if __name__ == "__main__":
    main()
