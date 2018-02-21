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
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

# please specify the location of the data here
DATA_FOLDER = '/homeappl/home/sayfull1/NYC'

parser = argparse.ArgumentParser(description='Loading Book Corpus dataset')
parser.add_argument('-data-folder', type=str, default="/homeappl/home/sayfull1/NYC", help='Folder, where book corpus tar is')
parser.add_argument('-include-syntax', type=bool, default=True, help='whether to use punctuation in vocabulary or not')
args = parser.parse_args()
print args


class BookCorpus(object):
    
    def __init__(self, data_folder, small_dataset=False, is_wiki_train=False, stop_words=True,train_sentences=None,max_voc=None):
        
	self.remove_stop_words = stop_words
	if not small_dataset:
        	self.train_sentences = 30000
		self.max_voc = 15000
	else:
		self.train_sentences = 5000
		self.max_voc = 5000

        if train_sentences is not None:
		self.train_sentences = train_sentences

	if max_voc is not None:
		self.max_voc = max_voc
		

        self.valid_sentences = 20
        self.test_sentences = 20
        
        self.word_embed_dim = 200
        self.max_num_words = 30
        
        self.Xtrain = []
        self.Xvalid = []
        self.Xtest = []
        
	self.embed_matrix = None
        self.vocabulary = []
        self.vocabulary_set = {}
        self.vocab_to_ind = {}
        
        self.data_folder = data_folder
   
	self.wnl = WordNetLemmatizer()     
	
	if is_wiki_train:
	    data_folder_wiki = "/wrk/sayfull1/NYC/"
	    self.Xtrain, self.XtrainLens = self.get_wikitrain_data(data_folder_wiki+'/wikitext-2/wiki.train.tokens', self.train_sentences)
	    self.Xvalid, self.XvalidLens = self.get_wikitrain_data(data_folder_wiki+'/wikitext-2/wiki.valid.tokens', self.valid_sentences) 
	    self.Xtest, self.XtestLens = self.get_wikitrain_data(data_folder_wiki+'/wikitext-2/wiki.test.tokens', self.test_sentences) 	

	else:
	    self.split_data()

	print type(self.Xtrain)
	print np.shape(self.Xtrain)

        self.find_vocabulary()
	if "<unk>" not  in self.vocabulary_set:
	    print '<unk> is missing'
	self.get_glove_embedding()
	#print "Last voc words", self.vocabulary[-50:-1]

    def get_wikitrain_data(self, full_filename,max_sentences):
    	
	'''
		Xtrain - the list of char sentences
		Lens - binary vector where each sentence in the same article has the same value, the opposite
		from the previous article. E.g. 3 sentences from 1 article
		then 2 sentences, then 3 sentences, etc: [0 0 0 1 1 0 0 0 1 1 1]
	'''	

	Xtrain = []
    	Lens = []
    	f = open(full_filename,'rb')
    	flag = 0
	flag_article_continues = False
	num_articles = 0
	
    	for i, row in enumerate(f.readlines()):
            row = row.decode("utf-8")
            if row.strip().startswith('='):
		flag_article_continues = False
                continue

            row = row.replace('\n','')
            sentences = row.split(".")
            sentences = list(filter(lambda x: x.strip(), sentences))
	    if not len(sentences):
		continue

            Xtrain = Xtrain + sentences
            s_len = len(list(sentences))
	    if not flag_article_continues: #new article
            	flag = not flag
		num_articles+=1

	    Lens.extend([int(flag)]*s_len)
	    flag_article_continues = True
	    if len(Xtrain) > max_sentences:
		break

	print 'number of various articles:', num_articles
        return Xtrain[0:max_sentences], Lens[0:max_sentences]
	
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
                		self.Xtrain.append(f.readline().rstrip().lower())
            		for i in range(self.valid_sentences):
                		self.Xvalid.append(f.readline().rstrip().lower())
            		for i in range(self.test_sentences):
                		self.Xtest.append(f.readline().rstrip().lower())
			f.close()
		break
	t.close()
	self.Xtrain_unprocessed = self.Xtrain
	print 'The len of Xtrain', len(self.Xtrain)
        print 'The len of Xvalid', len(self.Xvalid)
        print 'The len of Xtest', len(self.Xtest)
        
    def find_vocabulary(self):
        
	wnl = WordNetLemmatizer()
        # makes the lowercasing by default
        self.count_vect = CountVectorizer(decode_error='replace', max_features=self.max_voc)       
        self.count_vect.fit_transform(self.Xtrain).toarray()
        
        self.vocabulary = self.count_vect.get_feature_names()
	self.vocabulary_set = set(self.vocabulary)
	#if '<unk>' not in self.vocabulary:
	#	self.vocabulary.append('<unk>')
	#self.vocabulary = map(wnl.lemmatize, self.vocabulary)
	if self.remove_stop_words:
		for word in stopWords:
			if word in self.vocabulary_set:
				self.vocabulary.remove(word)

	if args.include_syntax:
		self.vocabulary = list(filter(lambda x: re.search(r"^[a-z,!\?]+$", x), self.vocabulary))
		self.vocabulary.extend(['!','?',','])
	else:
		self.vocabulary = list(filter(lambda x: re.search(r"^[a-z]+$", x), self.vocabulary))
		
	print "The length of the original vocabulary is", len(self.vocabulary)
	
	self.vocabulary = list(set(map(lambda x: self.wnl.lemmatize(x,pos='v'), self.vocabulary)))
	self.vocabulary = list(set(map(lambda x: self.wnl.lemmatize(x,pos='n'), self.vocabulary)))	

	self.vocabulary_set = set(self.vocabulary)	
	print "The length of the lemmatized vocabulary is", len(self.vocabulary_set)
        self.vocab_to_ind = dict({word:i for i, word in enumerate(self.vocabulary)})
	print sorted(self.vocabulary)[0:200]
	print sorted(self.vocabulary)[-200:-1]
    
    def get_glove_embedding(self):

	glove = vocab.GloVe(name='6B', dim=200)
	#"from:https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb"
	self.embed_matrix = 0.01 * np.random.uniform(-1,1,size=(len(self.vocabulary), 200))
	glove_indices = dict({glove.stoi[word]:i for i, word in enumerate(self.vocabulary) if word in glove.stoi})
        self.embed_matrix[glove_indices.values(),:] = glove.vectors.numpy()[glove_indices.keys()]
	
	'''glove_min = np.min(glove.vectors.numpy()[glove_indices.keys()])
	glove_max = np.max(glove.vectors.numpy()[glove_indices.keys()])
	glove_mean = np.mean(glove.vectors.numpy()[glove_indices.keys()])	
	print 'Glove min-max-mean:', glove_min, '-', glove_max, ' - ', glove_mean
	'''

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
        if self.remove_stop_words:
	    word_list = list(set(word_list).difference(stopWords))
           
	word_list = map(lambda x: self.wnl.lemmatize(x, pos='v'), word_list)
 	word_list = map(lambda x: self.wnl.lemmatize(x, pos='n'), word_list)
 
        #word_list = list(map(self.replace_unknown, word_list)) # may be remove this line
	words = [w for w in word_list \
			if w in self.vocabulary_set][0:self.max_num_words]
	#print words
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
            if not len(found_word_vectors):
	        continue	    
            Xtrain_[k, range(len(found_word_vectors))] = found_word_vectors               
            seq_len.append(len(found_word_vectors))
            k+=1
	    #print self.Xtrain_unprocessed[i]
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
   
    
       
    def testGetMeanGlove(self):
        assert_array_equal(self.bc.get_mean_glove("Amazing day today"), \
                           self.bc.get_mean_glove("Amazing today day"))
   
        #self.bc.get_mean_glove("Amazing day")

   
    def testGetBow(self):
        assert_array_equal(self.bc.get_bow("Amazing day today"), \
                           self.bc.get_bow("Amazing today day"))    
        #self.bc.get_bow("Amazing day today")
    '''
    
    def testGetDataWithIndices(self):
	#print self.bc.vocabulary
	
	data, seq_len, ind = self.bc.get_data_with_word_indices(self.bc.Xtrain)
	'''for k, i in enumerate(ind):
		print self.bc.Xtrain[i]
		print data[k]
		print seq_len[k]
	'''
	#for i, row in enumerate(data):
        #    print row, self.Xtrain_unprocessed[i]
	#print data
	#print seq_len
	#print np.shape(data)
	#print np.shape(seq_len)
    
  
if __name__ == '__main__':
    unittest.main(verbosity=2)        
	
