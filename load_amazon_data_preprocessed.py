# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 13:48:45 2017
Load and preprocess amazon sentiment datasent
http://pfliu.com/paper/adv-mtl.html
@author: Luiza
"""

import re, os
import numpy as np
import csv
import torchtext.vocab as vocab
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
import unittest
from nltk.tokenize import word_tokenize
from scipy import spatial

# please specify the location of the data here
DATA_FOLDER = '/wrk/sayfull1/NYC/mtl-dataset/mtl-dataset/'

class AmazonSentimentData(object):
    
    ''' The class for loading and preprocessing Amazon Review data for PyTorch
    check: load_one_dataset_variable_length function
    Data is available: http://pfliu.com/paper/adv-mtl.html
    '''
    
    def __init__(self, data_folder ='./mtl-dataset/ml-dataset/', max_num_words=800, dataset_name='baby', vocabulary_all_datasets=False):
        self.data_folder = data_folder
    
        if vocabulary_all_datasets:
            self.get_vocabulary()
        else:
            self.get_vocabulary_one_dataset(dataset_name=dataset_name)
    
        self.num_words = max_num_words
        self.embed_dim = 200
        self.get_glove_embedding()        
        self.max_num_words = max_num_words
   
    def clean_str(self, string):

            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

    def get_raw_data(self, full_filename):
        
        data_train = open(full_filename, 'r')
        reader = csv.reader(data_train,delimiter='\t')
        X = []
        Y = []
        for i, row in enumerate(reader):
            try:
                descr = self.clean_str(row[1])
                X.append(descr)
                Y.append(int(row[0]))
            except Exception as e:
                print e
        return X, Y    
        
    def get_list_of_all_datasets(self):
    
        all_files_in_folder = [f for f in listdir(self.data_folder)\
                               if isfile(join(self.data_folder, f))]
        topics = set()
        for f in all_files_in_folder:
            topics.add(f.split('.')[0])
  
        return list(topics)

    def get_vocabulary_one_dataset(self, dataset_name):

        all_files_in_folder = [f for f in listdir(self.data_folder)\
                               if isfile(join(self.data_folder, f))]

        self.count_vect = CountVectorizer(decode_error='replace', max_features=60000)
        all_text = []
        number_datasets = 0
        for f in all_files_in_folder:
            data_set_type = f.split('.')[-1]
            topic = f.split('.')[0]
            if (data_set_type != 'train') or (topic!=dataset_name):
                continue
            X, Y = self.get_raw_data(self.data_folder + f)
            number_datasets+=1
            all_text.extend(X)

        self.count_vect.fit_transform(all_text).toarray()
        self.vocabulary = self.count_vect.get_feature_names()
        self.vocabulary_set = set(self.vocabulary)
        self.vocab_to_ind = {word:i for i, word in enumerate(self.vocabulary)}

    def get_vocabulary(self):
        
        ''' Builds the vocabulary for all the datasets
        '''
        
        all_files_in_folder = [f for f in listdir(self.data_folder)\
                               if isfile(join(self.data_folder, f))]
        
        self.count_vect = CountVectorizer(decode_error='replace', max_features=500000)
        all_text = []
        number_datasets = 0
        for f in all_files_in_folder:
            data_set_type = f.split('.')[-1] 
            if data_set_type != 'train':
                continue
            X, Y = self.get_raw_data(self.data_folder + f)
            number_datasets+=1
            all_text.extend(X)


        self.count_vect.fit_transform(all_text).toarray()
        self.vocabulary = self.count_vect.get_feature_names()
        self.vocabulary_set = set(self.vocabulary)
        self.vocab_to_ind = {word:i for i, word in enumerate(self.vocabulary)}
    
    def get_paragraph_matrix(self,Xtrain):

        ''' Returns word-vector representations, labels of text samples
        '''
        
        Xtrain_ = np.zeros((len(Xtrain), self.num_words, self.embed_dim))
        words_per_sentence = 0
        for i, x in enumerate(Xtrain):
            paragraph_word_list = np.zeros((self.num_words,self.embed_dim))
            word_list = word_tokenize(x) 
            found_word_vectors = [self.embed_matrix[self.vocab_to_ind[w]] for w in word_list if w in self.vocabulary_set][0:self.num_words]
            if found_word_vectors:
                paragraph_word_list[range(len(found_word_vectors))] = found_word_vectors  
                Xtrain_[i] = paragraph_word_list
                words_per_sentence+=len(paragraph_word_list)
            else:
                print i, x
    
        print 'The average length of the sentence is ', words_per_sentence/len(Xtrain)
        return Xtrain_

    def get_paragraph_matrix_variable_length(self, Xtrain, Ytrain):

        ''' Returns word-vector representations, labels and length of text samples
        '''
        
        Xtrain_ = np.zeros((len(Xtrain), self.max_num_words, self.embed_dim))
        words_per_sentence = 0
        seq_len = []
        for i, x in enumerate(Xtrain):
                word_list = word_tokenize(x)
                # done as in Pengfei Li code
                #word_list = x.replace("-"," ").replace("/"," ").replace("\'","").replace("."," ").split(" ")
                found_word_vectors = [self.embed_matrix[self.vocab_to_ind[w]] for w in word_list if w in self.vocabulary_set][0:self.max_num_words]
                Xtrain_[i,range(len(found_word_vectors))] = found_word_vectors
                text_len = len(found_word_vectors)
                words_per_sentence+=text_len
                seq_len.append(len(found_word_vectors))
               
        seq_len = np.array(seq_len)
        Ytrain = np.array(Ytrain)
        ind = np.argsort(np.array(seq_len))[::-1]
        ind = np.array(ind, dtype=np.int32)
        #print 'The average length of the sentence is ', words_per_sentence/len(Xtrain)
        return Xtrain_[ind,:,:], Ytrain[ind], seq_len[ind]


    def load_one_dataset_bow(self, filename, unlabelled = False):
        
        '''  Given:
                filename: e.g. "apparel", "baby"
             Returns:
                A tuple of Xtrain, Ytrain, Xtest, Ytest
                where X... is a list of strings and Y... is a label for those strings 
        '''
        
        Xtrain, Ytrain = self.get_raw_data(self.data_folder + filename + '.task.train')
        Xtest, Ytest = self.get_raw_data(self.data_folder + filename + '.task.test')
        
        if unlabelled:
            Xunlabel, Yunlabel = self.get_raw_data(self.data_folder + filename + '.task.unlabel','r')
        
        return self.count_vect.transform(Xtrain), Ytrain, self.count_vect.transform(Xtest), Ytest
    

    def load_one_dataset_variable_length(self, filename, unlabelled=True):

        ''' Loads variable length dataset of max_num_words length per sample
        initialized with 200d Glove embedding
        ''' 
        
        Xtrain, Ytrain = self.get_raw_data(self.data_folder + filename + '.task.train')
        Xtest, Ytest = self.get_raw_data(self.data_folder + filename + '.task.test')
        # takes 200 samples for the validation as in Pengfei split
        Xvalid,Yvalid = Xtrain[-200:],Ytrain[-200:]
        Xtrain, Ytrain = Xtrain[0:-200], Ytrain[0:-200]
        return self.get_paragraph_matrix_variable_length(Xtrain, Ytrain), self.get_paragraph_matrix_variable_length(Xvalid, Yvalid),\
        self.get_paragraph_matrix_variable_length(Xtest, Ytest) 

    def load_one_dataset(self, filename, unlabelled=True):
        
        ''' Loads fixed length dataset
        '''
        
        Xtrain, Ytrain = self.get_raw_data(self.data_folder + filename + '.task.train')
        Xtest, Ytest = self.get_raw_data(self.data_folder + filename + '.task.test')
        return self.get_paragraph_matrix(Xtrain), Ytrain, self.get_paragraph_matrix(Xtest), Ytest

    def get_glove_embedding(self):
        
        #"from:https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb"
        glove = vocab.GloVe(name='6B', dim=200)
        self.embed_matrix = 0.01*np.random.rand(len(self.vocabulary),200)
        glove_indices = dict({glove.stoi[word]:i for i, word in enumerate(self.vocabulary) if word in glove.stoi})
        self.embed_matrix[glove_indices.values(),:] = glove.vectors.numpy()[glove_indices.keys()]
        
class TestAmazonSentimentData(unittest.TestCase):

    def setUp(self):
        self.amazon_data = AmazonSentimentData(DATA_FOLDER)
     
    def testgetListDatasets(self):
        print 'The list of datasets'
        print self.amazon_data.get_list_of_all_datasets()    
 
    def testLoadOneDataset(self):
        (Xtrain, Ytrain, Lentrain ),(Xvalid, Yvalid, Lenvalid), (Xtest, Ytest, Lentest) = self.amazon_data.load_one_dataset_variable_length("baby")
        print np.shape(Xtrain), np.shape(Lentrain)    

if __name__ == '__main__':
    unittest.main(verbosity=2)
        