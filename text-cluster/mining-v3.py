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

from project_consts import *

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

        def train_extractor_from_lines(self, train_lines, labels, test_lines):
                self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_features=DISTINCT_WORDS_CNT)
                self.vectorizer.fit(train_lines + test_lines)

                pass

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

        def train_extractor(self, full = False):

                if not full:
                        train_lines = file2lines('../data/train_lite.csv')
                        labels = file2labels('../data/train_lite.csv')
                        test_lines = file2lines('../data/test_lite.csv')
                else:
                        train_lines = file2lines('../data/train.csv')
                        labels = file2labels('../data/train.csv')
                        test_lines = file2lines('../data/test.csv')

                self.train_extractor_from_lines(train_lines, labels, test_lines)

                pass

        def lines2words(self, lines):
                self.tokenizer = self.vectorizer.build_tokenizer()

                return [self.tokenizer(line) for line in lines]

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

class Learner:

        # models = [(model, select_vec), (model, select_vec), ...]
        models = []

        def train(self, train_data, train_labels, algo):
                print "Training..."
                # VERSION 1
                # train a single model for each label dimension
                self.models = []
                label_ver = zip(*train_labels)
                for labels in label_ver:
                        select_vec = self.select_feature_from_single_labels(train_data, labels, FEATURE_SELECTION_CNT)
                        select_data = train_data.take(select_vec, axis=1)
                        model = self.train_single_model(select_data, labels, algo)
                        self.models.append((model, select_vec))

                return self.models

        def predict(self, test_data):
                print ""
                print "Predicting"
                fused_prediction = []
                for model, select_vec in self.models:
                        select_test_data = test_data.take(select_vec, axis=1)
                        fused_prediction.append(model.predict(select_test_data))

                return np.array(fused_prediction).T

        def evaluate(self, prediction, test_labels):
                print ""
                print "Evaluating"
                """
                Evaluate the result. 
                Print stats description and return the error series
                """
                n_pred = np.float32(np.array(prediction))
                n_labels = np.float32(np.array(test_labels))

                error = DataFrame(np.abs(n_labels - n_pred), columns = label_desc)

                column_error = error.apply(lambda x: np.sqrt(x.dot(x)), axis=0) / np.sqrt(len(error))
                print column_error
                print "The global RMSE is ", np.sum(column_error) / len(column_error)
                sum_s = column_error[0:4].mean()
                sum_w = column_error[5:8].mean()
                sum_k = column_error[9:].mean()
                print "Sentimental :", sum_s
                print "When        :", sum_w
                print "Kind        :", sum_k

                return column_error

        def train_single_model(self, train_data, train_labels, algo):
                print ".",
                """
                Train the model for a single label dimension
                """
                if algo == 'svr_rbf':
                        """
                        SVM regression, RBF kernel
                        """
                        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
                        svr_rbf.fit(train_data, train_labels)
                        return svr_rbf

                if algo == 'svr_lin':
                        """
                        SVM regression, linear
                        """
                        svr_lin = SVR(kernel='linear')
                        svr_lin.fit(train_data, train_labels)
                        return svr_lin

                if algo == 'ridge':
                        """
                        Ridge regression
                        """
                        clf = Ridge(alpha = 0.5)
                        clf.fit(train_data, train_labels)
                        return clf

                # No hit algorithm
                print "unimplemented model type"
                return None

        def select_feature_from_single_labels(self, train_data, single_labels, feature_num):
                # VERSION 1, correlation
                
                select_vec = []
                corrs = []

                for i in range(train_data.shape[1]):
                        v1 = np.squeeze(np.array(train_data.take([i], axis=1))) * 2 - 1
                        v2 = np.array(np.float32(single_labels)) * 2 - 1
                        corr = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
                        corrs.append(np.abs(np.dot(v1,v2) / np.sqrt(np.dot(v1,v1) * np.dot(v2,v2))))

                select_vec = np.array(corrs).argsort()[-feature_num:]

                return select_vec
                """
                # VERSION 2, SelectKBest
                sk = SelectKBest()
                sk.k = FEATURE_SELECTION_CNT
                sk.fit(train_data, single_labels)
                select_vec = [i for i in range(len(sk.get_support())) if sk.get_support()[i]]
                """

                return sk.get_support()
                

### Global Variables
# feature extractor
FE = FeatureExtractor()
# stemmer
st = LancasterStemmer()
# learner
L  = Learner()

def stem(word):
        if word in convert_dict:
                return convert_dict[word]
        else:
                return word

### Methods
# main routine
def routine_test():

        train_filename = '../data/train.csv'
        test_filename = '../data/test.csv'

        FE.load_vectorizer()
        train_lines = file2lines(train_filename)
        test_lines = file2lines(test_filename)
        train_labels = file2labels(train_filename)

        train_features = FE.lines2features(train_lines, use_tense = USE_TENSE)
        test_features = FE.lines2features(test_lines, use_tense = USE_TENSE)

        train_features.to_csv('../data/train_features.csv')
        test_features.to_csv('../data/test_features.csv')

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
        train_lines = file2lines('../data/train.csv')
        test_lines = file2lines('../data/test.csv')
        train_labels = file2labels('../data/train.csv')
        test_ids = file2ids('../data/test.csv')

        print "Get features"

        train_features = FE.lines2features(train_lines, use_tense = USE_TENSE)
        test_features = FE.lines2features(test_lines, use_tense = USE_TENSE)

        train_features.to_csv('../data/train_features.csv')
        test_features.to_csv('../data/test_features.csv')

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