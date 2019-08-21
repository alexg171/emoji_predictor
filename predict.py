import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from pandas import DataFrame
from sklearn import metrics
import numpy
import re
import sys


# usdata = '1k_tweets.txt'
# uslabel = '1k_labels.txt'
# usdata = '5k_tweets.txt'
# uslabel = '5k_labels.txt'
# usdata = '10k_tweets.txt'
# uslabel = '10k_labels.txt'
usdata = '20k_tweets.txt'
uslabel = '20k_labels.txt'
# usdata = '30k_tweets.txt'
# uslabel = '30k_labels.txt'
# usdata = '40k_tweets.txt'
# uslabel = '40k_labels.txt'
# usdata = 'tweets_us.json.text'
# uslabel = 'tweets_us.json.labels'


class train:
    def __init__(self,trainingdocs,traininglabels):
        print("Opening docs...")
        self.docs = open(trainingdocs)
        self.labels = open(traininglabels)
        
        print("Vectorizing data...")
        self.label_vect = CountVectorizer(token_pattern ='[0-9]{1,2}')
        self.label_tdm = self.label_vect.fit_transform(self.labels)
    
        self.data_vect = CountVectorizer(stop_words='english',lowercase='True')
        self.data_tdm = self.data_vect.fit_transform(self.docs)
       
        print("Term document matrices to arrays...")
        self.data_A = self.data_tdm.A
        self.label_A = self.label_tdm.A

        print("Initializing vocab lists...")
        self.data_vocab = self.data_vect.vocabulary_
        self.label_vocab = self.label_vect.vocabulary_
        
        print("Matrix calculations...")
        self.doc_sums = self.data_A.sum(axis=1)
        self.label_sums = self.label_tdm.toarray().sum(axis=0) 
        self.V = len(self.data_vocab)
        print("Ready to classify...")

    def countwc(self,word,klass):
        if word not in self.data_vocab:
            return 0
        return sum(self.data_A[:,self.data_vocab[word]]*self.label_A[:,self.label_vocab[klass]])

    def countc(self,klass):
        return sum(self.doc_sums*self.label_A[:,self.label_vocab[klass]])

    def word_class_prob(self,word,klass):
        return float(self.countwc(word,klass) + 1) / float(self.countc(klass) + self.V + 1)

    def label_prob(self,klass):
        return  float(self.label_sums[self.label_vocab[klass]])/float(sum(self.label_sums))

    def doc_class_prob(self,doc,klass):
        pwc = 1
        for word in doc:
            pwc = pwc*self.word_class_prob(word,klass)
        return pwc


    def classify(self,doc):
        max_prob = -1
        result = (0,0)
        for klass in self.label_vocab:
            pwc = self.doc_class_prob(doc,klass)
            pc = self.label_prob(klass)
            p = pwc*pc
            if p > max_prob:
                max_prob = p
                result = (klass,p)
        return result

phrases = ['merry christmans', 'hapy 4th of july', 'i hate you', 'i love you']

def main():
    dat = open(usdata)
    lab = open(uslabel)
    t = train(usdata,uslabel)

    ############################ METHOD 1
    # if len(sys.argv) != 2:
    #     print "Incorrect number of arguments, try again"
    # arg = sys.argv[1]
    # tweet = re.sub("[^\w]", " ", arg.strip().lower()).split()
    # print tweet
    # print t.classify(tweet)
    ############################
    
    ############################ METHOD 2
    for phrase in phrases:
        tweet = re.sub("[^\w]", " ", phrase.strip().lower()).split()
        print ("%s ==> %s" % (phrase,t.classify(tweet)))
    ############################

if __name__ == "__main__":
    main()
