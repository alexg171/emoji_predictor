import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pandas import DataFrame
from sklearn import metrics
import numpy
import re
import time

# usdata = '1k_tweets.txt'
# uslabel = '1k_labels.txt'
# usdata = '5k_tweets.txt'
# uslabel = '5k_labels.txt'
# usdata = '10k_tweets.txt'
# uslabel = '10k_labels.txt'
# usdata = '20k_tweets.txt'
# uslabel = '20k_labels.txt'
# usdata = '30k_tweets.txt'
# uslabel = '30k_labels.txt'
# usdata = '40k_tweets.txt'
# uslabel = '40k_labels.txt'
# usdata = 'tweets_us.json.labels'
# uslabel = 'tweets_us.json.text'

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


    def classifyDoc(self,doc):
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

    def predict(self, testset):
        predicted = []
        #i = 0
        start = time.time()
        for doc in testset:
            #print i
            #i = i + 1
            res = self.classifyDoc(doc)
            klass = int(res[0].strip())
            predicted.append(klass)
        return predicted

def accuracy(predicted,target_test):
    matches = 0
    for x in range(len(predicted)):
        if(predicted[x] == target_test[x]):
            matches += 1

    return float(matches)/float(len(predicted)) #, elapsed, "seconds"

def test(usdata,uslabel):
    start_t = time.time()

    mynb_clf = train(usdata,uslabel)

    end_t = time.time()
    trainingtime = end_t - start_t

    usdata = open(usdata)
    uslabel = open(uslabel)

    target = []
    for line in uslabel.readlines():
        target.append(int(line.strip()))
    
    test = []
    for line in usdata.readlines():
        line_list = re.sub("[^\w]", " ", line.strip().lower()).split()
        test.append(line_list)


    print "target",len(target)
    print "test",len(test)
    print mynb_clf
    print usdata
    start_t = time.time()

    for i in range(10):
        start = i*100
        end = start + 100
        print "Range: ", start, end
        docs_test = test[start:end]
        target_test = target[start:end]
        predicted = mynb_clf.predict(docs_test)
        print "accuracy: ", accuracy(predicted,target_test)
    
    end_t = time.time()
    elapsed = end_t - start_t
    print "training time", trainingtime
    print "elapsed time", elapsed
    
def main():
    datalabelpairs = [ ('5k_tweets.txt','5k_labels.txt'),('10k_tweets.txt','10k_labels.txt'),('20k_tweets.txt','20k_labels.txt'),('30k_tweets.txt','30k_labels.txt'),('40k_tweets.txt','40k_labels.txt'),('tweets_us.json.text','tweets_us.json.labels')]

    for set in datalabelpairs:
        print "===============================", set
        test(set[0],set[1])

   
if __name__ == "__main__":
    main()
