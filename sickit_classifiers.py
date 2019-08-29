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


def test(usdata,uslabel):
    a = open(usdata, encoding="utf8")
    b = open(uslabel, encoding="utf8")
    c = open(usdata, encoding="utf8")

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(a)

    print("***********************")
    print("Count")

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    target = []
    for line in b.readlines():
        target.append(int(line.strip()))

    start_t = time.time()
    
    # classifiers
    # clf = MultinomialNB().fit(X_train_tfidf, target)
    # clf = SGDClassifier().fit(X_train_tfidf, target)
    # clf = svm.SVC().fit(X_train_tfidf, target)
    # clf = LogisticRegression().fit(X_train_tfidf, target)
    # clf = KNeighborsClassifier().fit(X_train_tfidf, target)
    # clf = DecisionTreeClassifier().fit(X_train_tfidf, target)
    clf = MLPClassifier().fit(X_train_tfidf, target)
    
    end_t = time.time()
    trainingtime = end_t - start_t

    test = []
    for line in c.readlines():
        test.append(line.strip())

    print(clf)
    print(usdata)
    start_t = time.time()

    for i in range(10):
        start = i*100
        end = start + 100
        print("Range: ", start, end)
        docs_test = test[start:end]
        target_test = target[start:end]
        X_new_counts = count_vect.transform(docs_test)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)
        print("accuracy: ", np.mean(predicted == target_test))
    
    end_t = time.time()
    elapsed = end_t - start_t
    print("training time", trainingtime)
    print("elapsed time", elapsed)

def main():
    if len(sys.argv) < 3:
        print("Wrong number of arguments")
        exit()

    data = sys.argv[1]
    labels = sys.argv[2]
    test(data,labels)


if __name__ == "__main__":
    main()
