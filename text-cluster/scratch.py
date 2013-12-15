import csv
import itertools
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import numpy as np
from pandas import *
import matplotlib.pyplot as plt

import nltk
from nltk.stem.lancaster import LancasterStemmer


### Tunning
DISTINCT_WORDS_CNT = 700
FEATURE_SELECTION_CNT = 400
USE_TENSE = False

### Global Utilities
# tokenizer
def tokenizer(text):
    tok = nltk.tokenize.RegexpTokenizer(r'\w{3,}')
    stopwords = nltk.corpus.stopwords.words('english')
    return [stem(w.lower()) for w in tok.tokenize(text) if w.lower() not in stopwords]

### Class
class FeatureExtractor:

    vectorizer = None
    feature_names = None
    feature_matrix = None

    def train_extractor_from_lines(self, lines):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_features=DISTINCT_WORDS_CNT)
        self.vectorizer.fit(lines)

        pass

    def train_extractor(self, full = False):
        lines = file2lines('data/train_lite.csv')
        self.train_extractor_from_lines(lines)

        pass

    def get_lines_distance_matrix(self, lines):

        mat = np.zeros((len(lines), len(lines)))
        feature_df = self.lines2features(lines)
        for index1, row1 in feature_df.iterrows():
            for index2, row2 in feature_df.iterrows():
                mat[index1, index2] = self.feature_distance(row1, row2)

        return mat

    def lines2features(self, lines, use_tense = False):
        """
        returns DataFrame(feature_matrix, feature_name)

        ['word_rainny', 'word_'sunny'],
        array([
                [1, 0.4, 0.2],
                [0.2, 1, 0.2],
        ])
        """
        self.feature_names = []
        self.feature_matrix = None

        # tf-idf features
        data = self.vectorizer.transform(lines).toarray()

        self.feature_names = self.vectorizer.get_feature_names()
        self.feature_matrix = data

        # additional features
        add_features = []
        important_words = ['sunny', 'wind', 'humid', 'hot', 'cold', 'dry', 'ice', 'rain', 'snow', 'tornado', 'storm', 'hurricane']
        important_words = ['cloud', 'cold', 'dry', 'hot', 'humid', 'hurricane', 'ice', 'rain', 'snow', 'storm', 'sunny', 'tornado', 'wind']
        self.feature_names = self.feature_names + ['impt_words:' + word for word in important_words]
        if use_tense:
                self.feature_names = self.feature_names + ['past_tense_num', 'present_tense_num']

        all_words = self.lines2words(lines)
        for words in all_words:
                # important words
                important_words_ftr = [int(word in words) for word in important_words]
                add_features.append(important_words_ftr)

                # tense
                if use_tense:
                        tagz = zip(*nltk.pos_tag(nltk.word_tokenize(words)))[1]
                        past_num = len([v for v in tagz if v == 'VBD'])
                        present_num = len([v for v in tagz if v in ['VBP', 'VB']])

                        add_features.append([past_num, present_num])
    
        self.feature_matrix = np.hstack((self.feature_matrix, add_features))

        return DataFrame(self.feature_matrix, columns = self.feature_names)

    def feature_distance(self, feature_vector1, feature_vector2):
        # preliminary version
        return 1 - np.dot(feature_vector1, feature_vector2) / np.sqrt((np.dot(feature_vector1, feature_vector1) * np.dot(feature_vector2, feature_vector2)));

    def lines2words(self, lines):
        self.tokenizer = self.vectorizer.build_tokenizer()

        return [self.tokenizer(line) for line in lines]

    def load_vectorizer(self):
        input_file = open('../models/tfidf_vectorizer.pkl', 'rb')
        self.vectorizer = pickle.load(input_file)
        input_file.close()
        pass

    def save_vectorizer(self):
        output_file = open('../models/tfidf_vectorizer.pkl', 'wb')
        pickle.dump(self.vectorizer, output_file)
        output_file.close()
        pass

class HClust:

    breadscrum = {}

    def clust(self, mat, h_thre = 999999, n_thre = 99999):

        m = mat.copy()
        link = []

        # pre process
        for i in range(mat.shape[1]):
            mat[i,i] = 999999

        # iteration
        n_iter = np.min((mat.shape[1]-1, n_thre))

        for it in range(n_iter):
            n = m.shape[1]
            c1 = m.argmin() % m.shape[1]
            c2 = int(np.floor(m.argmin() / m.shape[1]))
            link.append([mat.shape[1]+it, c1, c2, m.min()])
            self.breadscrum[c1] = mat.shape[1]+it
            self.breadscrum[c2] = mat.shape[1]+it

            if link[-1][-1] > h_thre:
                break
            # combine
            m = np.concatenate((m, np.zeros((m.shape[0],1))), axis=1)
            m = np.concatenate((m, np.zeros((1, m.shape[1]))), axis=0)
            for i in range(m.shape[1]):
                if i == c1 or i == c2:
                    m[i,m.shape[1]-1] = 999999
                else:
                    m[i,m.shape[1]-1] = np.min([m[i,c1], m[i,c2]])
            for i in range(m.shape[1]):
                m[m.shape[1]-1,i] = m[i,m.shape[1]-1]
            m[m.shape[1]-1, m.shape[1]-1] = 999999

            # block
            for i in range(m.shape[1]):
                m[i,c1] = m[i,c2] = 999999
                m[c1,i] = m[c2,i] = 999999
        
        return link
        
    def link2cluster(self, link, n):
        cluster = {}
        for i in range(n+len(link)):
            ind = i
            while ind in self.breadscrum.keys():
                ind = self.breadscrum[ind]
            cluster[i] = ind
        return cluster





### Global Variables
# feature extractor
FE = FeatureExtractor()
# stemmer
st = LancasterStemmer()
def stem(word):
    return word
# cluster
H = HClust()

### Methods
# main routine
def routine_test():

    train_filename = 'data/train_lite.csv'

    lines = file2lines(train_filename)
    FE.train_extractor()
    features = FE.lines2features(lines)
    mat = FE.get_lines_distance_matrix(lines)

    links = H.clust(mat, h_thre = 0.8)

    FE.load_vectorizer()
    train_lines = file2lines(train_filename)
    train_labels = file2labels(train_filename)

    train_features = FE.lines2features(train_lines, use_tense = USE_TENSE)

    train_features.to_csv('data/train_features.csv')

    for k in range(10):
            # cross validation
            train_ind = []
            test_ind = []
            for i in range(len(train_lines)):
                    if np.random.rand() > 0.9:
                            test_ind.append(i)
                    else:
                            train_ind.append(i)
            train_data = np.matrix(train_features.take(train_ind, axis=0))
            test_data = np.matrix(train_features.take(test_ind, axis=0))
            train_labels_l = []
            test_labels_l = []
            for i in range(len(train_lines)):
                    if i in train_ind:
                            train_labels_l.append(train_labels[i])
                    else:
                            test_labels_l.append(train_labels[i])

            print ""
            print "Ridge Regression"
            L.train(train_data, train_labels_l, 'ridge')
            prediction = L.predict(test_data)
            eval = L.evaluate(prediction, test_labels_l)

    print ""
    print "svr_Lin Regression"
    L.train(train_data, train_labels_l, 'svr_lin')
    prediction = L.predict(test_data)
    eval = L.evaluate(prediction, test_labels_l)

    print ""
    print "svr_rbf Regression"
    L.train(train_data, train_labels_l, 'svr_rbf')
    prediction = L.predict(test_data)
            
    eval = L.evaluate(prediction, test_labels_l)


    return eval

def routine_work():

    import time
    start_time = time.time()

    print "Read data"

    FE.load_vectorizer()
    train_lines = file2lines('data/train.csv')
    test_lines = file2lines('data/test.csv')
    train_labels = file2labels('data/train.csv')
    test_ids = file2ids('data/test.csv')

    print "Get features"

    train_features = FE.lines2features(train_lines, use_tense = USE_TENSE)
    test_features = FE.lines2features(test_lines, use_tense = USE_TENSE)

    train_features.to_csv('data/train_features.csv')
    test_features.to_csv('data/test_features.csv')

    # cross validation
    train_data = np.matrix(train_features)
    test_data = np.matrix(test_features)

    L.train(train_data, train_labels, 'ridge')
    prediction = L.predict(test_data)
    
    prediction_df = DataFrame(prediction, columns=label_name)
    prediction_df.insert(0, 'id', test_ids)
    prediction_df.to_csv('../submission/submit.csv', index=False, float_format='%.3f')

    # Calculate time
    print 'Execution time: ', time.time() - start_time, 'seconds.'

    pass

def file2lines(input_file):
    """
    returns [
            ['a', 'b', 'c'],
            [],
            []
    ]
    """
    tweets = []
    with open(input_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader, None) # skip the header
            for row in spamreader:
                    tweets.append(row[1])

    return tweets

def file2labels(input_file):
    """
    Reads the file and returns raw labels
    """
    labels = []
    with open(input_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader, None) # skip the header
            for row in spamreader:
                    labels.append(row[4:28])

    return labels

def file2ids(input_file):
    """
    Reads the ids in test data
    """
    ids = []
    with open(input_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader, None) # skip the header
            for row in spamreader:
                    ids.append(row[0])

    return ids