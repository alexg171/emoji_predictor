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

def filetoarray(file):
    temp = []
    for line in file.readlines():
        temp.append(line.strip())
    return temp

def train(data,labels):
    # read in data
    tweets = open(data, encoding="utf8")
    emojis = open(labels, encoding="utf8")

    # tokenize words using a term-doc matrix
    count_vect = CountVectorizer()
    term_doc_matrix = count_vect.fit_transform(tweets)

    # normalize tokens
    tfidf_transformer = TfidfTransformer()
    normalized_tdm = tfidf_transformer.fit_transform(term_doc_matrix)

    # put emojis into an array
    target = filetoarray(emojis)

    # timer
    start_t = time.time()
    
    # classifiers
    # clf = MultinomialNB().fit(normalized_tdm, target)
    # clf = SGDClassifier().fit(normalized_tdm, target)
    # clf = svm.SVC().fit(normalized_tdm, target)
    # clf = LogisticRegression().fit(normalized_tdm, target)
    # clf = KNeighborsClassifier().fit(normalized_tdm, target)
    # clf = DecisionTreeClassifier().fit(normalized_tdm, target)
    clf = MLPClassifier().fit(normalized_tdm, target)

    # store model
    joblib.dump(clf, 'out/test_classifier.joblib')
    
    # end timer to measure how long it took to train
    end_t = time.time()
    trainingtime = end_t - start_t

    return 'out/test_classifier.joblib'


def test(data, labels, clffile):
    # load data
    tweets = filetoarray(open(data, encoding="utf8"))
    emojis = filetoarray(open(labels, encoding="utf8"))

    test_vect = CountVectorizer()
    test_transformer = TfidfTransformer()

    test_tdm = test_vect.fit_transform(tweets)
    test_tidf = test_transformer.fit_transform(test_tdm)

    # load classifier
    clf = joblib.load(clffile)

    # k-fold cross validation
    for i in range(10):
        start = i*100
        end = start + 100

        print ("***** Iteration: %s *****" % i)
        print("Range: ", start, end)

        docs_test = tweets[start:end]
        target_test = emojis[start:end]
        test_tdm = test_vect.transform(docs_test)
        test_normalized_tdm = test_transformer.transform(test_tdm)
        predicted = clf.predict(test_normalized_tdm)

        print("Accuracy: ", np.mean(predicted == target_test))

def main():
    if len(sys.argv) < 3:
        print("Wrong number of arguments")
        exit()

    data = sys.argv[1]
    labels = sys.argv[2]
    
    #clf = train(data,labels)

    test(data,labels, 'out/test_classifier.joblib')




if __name__ == "__main__":
    main()
