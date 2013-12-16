import os
import random
import csv
import itertools
from collections import Counter
import pickle
import unicodedata

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
DISTINCT_WORDS_CNT = 100
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
    features = None

    def train_extractor_from_lines(self, lines):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_features=DISTINCT_WORDS_CNT)
        self.vectorizer.fit(lines)

        pass

    def train_extractor(self, full = False):
        lines = dir2lines_labels('data/train_lite.csv')
        self.train_extractor_from_lines(lines)

        pass

    def get_features_distance_matrix(self, feature_df):
        mat = np.zeros((len(feature_df), len(feature_df)))
        
        for index1, row1 in feature_df.iterrows():
            for index2, row2 in feature_df.iterrows():
                mat[index1, index2] = self.feature_distance(row1, row2)

        return mat

    def get_lines_distance_matrix(self, lines):
        
        feature_df = self.lines2features(lines)

        self.features = feature_df

        return self.get_features_distance_matrix(feature_df)

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
        input_file = open('models/tfidf_vectorizer.pkl', 'rb')
        self.vectorizer = pickle.load(input_file)
        input_file.close()
        pass

    def save_vectorizer(self):
        output_file = open('models/tfidf_vectorizer.pkl', 'wb')
        pickle.dump(self.vectorizer, output_file)
        output_file.close()
        pass

class HClust:

    def clust(self, mat, h_thre = 999999, n_clust = 99999):

        m = mat.copy()
        link = []

        # pre process
        for i in range(mat.shape[1]):
            m[i,i] = 999999

        # iteration
        n_iter = np.min((mat.shape[1]-1, mat.shape[1]-n_clust))

        for it in range(n_iter):
            n = m.shape[1]
            c1 = m.argmin() % m.shape[1]
            c2 = int(np.floor(m.argmin() / m.shape[1]))
            link.append([mat.shape[1]+it, c1, c2, m.min()])

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

    def link2cluster_list(self, link, n):
        N = len(link) + n
        raw = []
        for i in range(n):
            raw.append([i])

        for i in range(len(link)):
            to = link[i][0]
            from1 = link[i][1]
            from2 = link[i][2]
            raw.append(raw[from1]+raw[from2])
            raw[from1] = []
            raw[from2] = []

        return [r for r in raw if r]

    def entropy(self, cluster_list, label_list):
        rst = 0
        for cluster in cluster_list:
            labels = [label_list[i] for i in cluster]
            c = Counter(labels)
            arr = zip(*c.items())[1]
            ent = 0
            for a in arr:
                p = float(a) / len(cluster)
                ent += -p * np.log(p)
            rst += ent
        return rst / len(cluster_list)

    def F_standard(self, cluster_list, label_list):
        
        n_f = {}
        n_r = {}
        r = {}
        p = {}
        # inverse index
        for i in range(len(cluster_list)):
            for j in range(len(cluster_list[i])):
                item = cluster_list[i][j]
                label_cur = label_list[item]
                if not label_cur in n_f.keys():
                    n_f[label_cur] = {}
                if not i in n_f[label_cur].keys():
                    n_f[label_cur][i] = 0
                n_f[label_cur][i] += 1
                if not i in n_r.keys():
                    n_r[i] = {}
                if not label_cur in n_r[i].keys():
                    n_r[i][label_cur] = 0
                n_r[i][label_cur] += 1

        for i in set(label_list):
            r[i] = {}
            p[i] = {}
            for j in range(len(cluster_list)):
                if j in n_f[i]:
                    v = n_f[i][j]
                else:
                    v = 0
                r[i][j] = float(v) / np.sum(n_f[i].values())
                p[i][j] = float(v) / np.sum(n_r[j].values())
        n = sum([sum(ni.values()) for ni in n_f.values()])
        
        F = {}
        for i in set(label_list):
            F[i] = {}
            for j in range(len(cluster_list)):
                if (r[i][j] + p[i][j]) == 0:
                    F[i][j] = 0
                else:
                    F[i][j] = 2*r[i][j]*p[i][j]/(r[i][j]+p[i][j])

        F_m = {}
        for i in set(label_list):
            tmp = []
            for j in range(len(cluster_list)):
                tmp.append(F[i][j])

            F_m[i] = np.max(tmp)

        F_m = {}
        for i in label_list:
            F_m[i] = {}
            F_m[i] = np.max([F[i][j] for j in range(len(cluster_list))])

        F_score = sum([float(sum(n_f[i].values()))/n*F_m[i] for i in set(label_list)])

        return F_score

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
# 1st clustering
def pre_clustering():

    train_filename = '/home/solesschong/Workspace/HW/text-cluster/data'

    lines_labels = dir2lines_labels(train_filename)
    random.shuffle(lines_labels)
    lines = zip(*lines_labels)[0]
    labels = zip(*lines_labels)[1]
    
    #FE.train_extractor_from_lines(lines[1:200])
    FE.load_vectorizer()
    
    N = len(lines)
    seg = int(np.ceil(float(N)/400))
    pre_cluster = []
    mat = FE.get_lines_distance_matrix(lines[1:2])
    centers = np.zeros((0, len(FE.features.columns)))
    for i in range(seg):
        upper = np.min(((i+1)*400, N))
        block_lines = lines[i*400:upper]
        mat = FE.get_lines_distance_matrix(block_lines)
        link = H.clust(mat, n_clust=50)
        cluster_list = H.link2cluster_list(link, upper-i*400)
        # generate center for each pre_cluster
        for j in range(len(cluster_list)):
            center = FE.features.iloc[cluster_list[j]].mean()
            centers = np.vstack((centers, np.array(center)))
        pre_cluster.extend(cluster_list)

    # Aggregate
    features2 = DataFrame(centers, columns=FE.features.columns)
    mat2 = FE.get_lines_distance_matrix(features2)
    link = H.clust(mat2, n_clust=30)
    cluster_list2 = H.link2cluster_list(link, mat2.shape[1])

    cluster_list_total = []
    for i in range(len(cluster_list2)):
        cluster_list_total.append([])
        for j in cluster_list2[i]:
            cluster_list_total[i].extend(pre_cluster[j])

    print H.entropy(cluster_list_total, labels)

    cluster_label_result = []
    for cluster in cluster_list_total:
        cluster_label_result.append([labels[i] for i in cluster])

    output_file = open('results/cluster_label_result_30.pkl', 'wb')        
    pickle.dump(cluster_label_result, output_file)

    # Aggregate
    features2 = DataFrame(centers, columns=FE.features.columns)
    mat2 = FE.get_lines_distance_matrix(features2)
    link = H.clust(mat2, n_clust=50)
    cluster_list2 = H.link2cluster_list(link, mat2.shape[1])

    cluster_list_total = []
    for i in range(len(cluster_list2)):
        cluster_list_total.append([])
        for j in cluster_list2[i]:
            cluster_list_total[i].extend(pre_cluster[j])

    print H.entropy(cluster_list_total, labels)

    cluster_label_result = []
    for cluster in cluster_list_total:
        cluster_label_result.append([labels[i] for i in cluster])

    output_file = open('results/cluster_label_result_50.pkl', 'wb')        
    pickle.dump(cluster_label_result, output_file)

    pass

# main routine
def routine_test():

    train_filename = 'data/train_lite.csv'

    lines_labels = dir2lines_labels(train_filename)
    lines = zip(*lines_labels)[0]
    labels = zip(*lines_labels)[1]

    FE.train_extractor()
    features = FE.lines2features(lines)
    mat = FE.get_lines_distance_matrix(lines)

    links = H.clust(mat, h_thre = 0.8)

    FE.load_vectorizer()
    train_lines = dir2lines_labels(train_filename)
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

def dir2lines_labels(input_dir):
    """
    returns [
            ['a', 'b', 'c'],
            [],
            []
    ]
    """
    lines_labels = []
    dirs = [d for d in os.listdir(input_dir) if not os.path.isfile(os.path.join(input_dir, d))]
    for dd in dirs:
        d = os.path.join(input_dir, dd)
        files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for filename in files:
            with open(filename) as f:
                file_lines = f.readlines()
            file_lines = [unicode(l, errors='replace') for l in file_lines]
            line = ""
            for l in file_lines:
                line += l
            lines_labels.extend([(line, dd)])

    return lines_labels

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