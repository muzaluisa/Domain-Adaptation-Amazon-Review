import tarfile,os
import sys
import re, os
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer
import unittest
import argparse
import torch
import torchtext.vocab as vocab
from nltk.tokenize import word_tokenize
from numpy.testing import assert_array_equal

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

# please specify the location of the data here
DATA_FOLDER = '/homeappl/home/sayfull1/NYC'

parser = argparse.ArgumentParser(description='Loading Book Corpus dataset')
parser.add_argument('-data-folder', type=str, default="/homeappl/home/sayfull1/NYC", help='Folder, where book corpus tar is')
args = parser.parse_args()
print args


class BookCorpus(object):
    
    def __init__(self, data_folder, small_dataset=True):
        
	self.remove_stop_words = True
	if not small_dataset:
        	self.train_sentences = 60000
		self.max_voc = 5000
	else:
		self.train_sentences = 10000
		self.max_voc = 2000

        self.valid_sentences = 20
        self.test_sentences = 20
        
        self.word_embed_dim = 200
        self.max_num_words = 50
        
        self.Xtrain = []
        self.Xvalid = []
        self.Xtest = []
        
	self.embed_matrix = None
        self.vocabulary = []
        self.vocabulary_set = {}
        self.vocab_to_ind = {}
        
        self.data_folder = data_folder
        
        self.split_data()
        self.find_vocabulary()
	self.get_glove_embedding()

    def split_data1(self):
        
        os.chdir(self.data_folder)
        tar = tarfile.open("books_in_sentences.tar")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            for i in range(self.train_sentences):
                self.Xtrain.append(f.readline().rstrip())
            for i in range(self.valid_sentences):
                self.Xvalid.append(f.readline().rstrip())
            for i in range(self.test_sentences):
                self.Xtest.append(f.readline().rstrip())
 	    #sys.exit()
	    break
	tar.close()
	print 'The len of Xtrain', len(self.Xtrain)        
	print 'The len of Xvalid', len(self.Xvalid)
	print 'The len of Xtest', len(self.Xtest)


    def split_data(self):
	
	t = tarfile.open(self.data_folder + '/books_in_sentences.tar', 'r')
	file_names = []
	for member_info in t.getmembers():
    		file_names.append(member_info.name)

	for filename in file_names:
    		try:
        		f = t.extractfile(filename)
    		except KeyError:
        		print 'ERROR: Did not find %s in tar archive' % filename
    		else:
			for i in range(self.train_sentences):
                		self.Xtrain.append(f.readline().rstrip())
            		for i in range(self.valid_sentences):
                		self.Xvalid.append(f.readline().rstrip())
            		for i in range(self.test_sentences):
                		self.Xtest.append(f.readline().rstrip())
			f.close()
		break
	t.close()
	print 'The len of Xtrain', len(self.Xtrain)
        print 'The len of Xvalid', len(self.Xvalid)
        print 'The len of Xtest', len(self.Xtest)
        
    def find_vocabulary(self):
        
        self.count_vect = CountVectorizer(decode_error='replace', max_features=self.max_voc)       
        self.count_vect.fit_transform(self.Xtrain).toarray()
        
        self.vocabulary = map(str,self.count_vect.get_feature_names())
	if self.remove_stop_words:
		for word in self.vocabulary:
			if word in stopWords:
				self.vocabulary.remove(word)	
        self.vocabulary_set = set(self.vocabulary)
        self.vocab_to_ind = dict({word:i for i, word in enumerate(self.vocabulary)})
    
    def get_glove_embedding(self):

	glove = vocab.GloVe(name='6B', dim=200)
	#"from:https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb"
	self.embed_matrix = 0.01 * np.random.rand(len(self.vocabulary), 200)
	glove_indices = dict({glove.stoi[word]:i for i, word in enumerate(self.vocabulary) if word in glove.stoi})
        self.embed_matrix[glove_indices.values(),:] = glove.vectors.numpy()[glove_indices.keys()]
	
    def replace_unknown(self, a):
        
	if a in self.vocabulary_set:
	    return a
	return "<UNK>"                
      
    def get_mean_glove(self, sentence):

        word_list = word_tokenize(sentence)
        word_list = list(map(self.replace_unknown, word_list))
        word_idx = [self.vocab_to_ind[w] for w in word_list \
                    if w in self.vocabulary_set][0:self.max_num_words]
        mean_glove = np.mean(self.embed_matrix[word_idx, :])
        return mean_glove
    
    def get_bow(self, sentence):
        
        word_list = word_tokenize(sentence)
        word_list = list(map(self.replace_unknown, word_list))
        vector = np.zeros(len(self.vocabulary_set))
        word_idx = [self.vocab_to_ind[w] for w in word_list \
                    if w in self.vocabulary_set][0:self.max_num_words]
        vector[word_idx] = 1
        return vector


    def get_word_indices(self, sentence):

        word_list = word_tokenize(sentence)
        word_list = list(map(self.replace_unknown, word_list))
        word_idx = [self.vocab_to_ind[w] for w in word_list \
                    if w in self.vocabulary_set][0:self.max_num_words]
        return word_idx        
        
    def get_data_with_word_indices(self, sentence_list):
    
        ''' Returns word-vector representations and length of text samples
            '''
        
        Xtrain_ = np.zeros((len(sentence_list), self.max_num_words))
        seq_len = []
        k = 0
	Xtrain_ind = []
        for i, x in enumerate(sentence_list):		
            found_word_vectors = self.get_word_indices(x)
            if len(found_word_vectors) < 3:
	        continue
            Xtrain_[k, range(len(found_word_vectors))] = found_word_vectors               
            seq_len.append(len(found_word_vectors))
            k+=1
	    Xtrain_ind.append(i)
            
        Xtrain_ = Xtrain_[0:k]       
        seq_len = np.array(seq_len)
        return Xtrain_, seq_len, Xtrain_ind
    
    def get_bow_data(self, sentence_list):
    
        ''' Returns word-vector representations and length of text samples
            '''
        
        Xtrain = np.zeros((len(sentence_list), len(self.vocabulary)))
        k = 0
        for i, x in enumerate(sentence_list):		
            vector_bow = self.get_bow(x)
            if len(vector_bow) < 3:
		continue
            Xtrain[k, range(len(vector_bow))] = vector_bow              
            k+=1   
        return Xtrain[0:k]
    
class TestBookCorpus(unittest.TestCase):

    def setUp(self):
        self.bc = BookCorpus(args.data_folder)
    
    '''def testSplitData(self):
	pass
        self.assertEqual(len(self.bc.Xtrain), self.bc.train_sentences)
        self.assertEqual(len(self.bc.Xvalid), self.bc.valid_sentences)
        self.assertEqual(len(self.bc.Xtest), self.bc.test_sentences)

    def testFindVocabulary(self):
        self.bc.find_vocabulary()
        print self.bc.vocabulary[0:10]
        print 'Number of words in voc:', len(self.bc.vocabulary)
   
    '''
       
    def testGetMeanGlove(self):
        assert_array_equal(self.bc.get_mean_glove("Amazing day today"), \
                           self.bc.get_mean_glove("Amazing today day"))
   
        #self.bc.get_mean_glove("Amazing day")

   
    def testGetBow(self):
        assert_array_equal(self.bc.get_bow("Amazing day today"), \
                           self.bc.get_bow("Amazing today day"))    
        #self.bc.get_bow("Amazing day today")
        
    def testGetDataWithIndices(self):
	data, seq_len, _ = self.bc.get_data_with_word_indices(self.bc.Xtrain)
	print np.shape(data)
	print np.shape(seq_len)
    
  
if __name__ == '__main__':
    unittest.main(verbosity=2)        
