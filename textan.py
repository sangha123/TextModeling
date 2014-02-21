from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy
import scipy as sp
import re
from pymongo import MongoClient

from BeautifulSoup import BeautifulSoup
import urllib2

from sklearn.cluster import KMeans
from HTMLParser import HTMLParser
import nltk.stem
import logging
from sklearn import decomposition
import unicodedata
from pandas import DataFrame, read_csv

global english_stemmer 
english_stemmer = nltk.stem.SnowballStemmer('english')

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


class StemmedTfidfVectorizer(TfidfVectorizer):
	
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class Textan(object):

	def tfidf(self, term, doc, docset): 
		tf = float( doc.count( term))/ sum( doc.count( w) for w in docset) 
		idf = math.log( float( len( docset))/( len([ doc for doc in docset if term in doc]))) 
		return tf * idf
	
 
	def retracttext(self):
		client = MongoClient()
		db = client.crunch
		self.overviewlist = []
		self.overview_tag = {}
		self.overview_name = {}
		self.comp_list = []
		cmpent=db.compdata
		cmp=cmpent.find()
		
		for cp in cmp:
			if cp.get('overview') and cp.get('tag_list'):
				s = MLStripper()
				s.feed(cp['overview'])
				overview=unicodedata.normalize('NFKD', 
            		s.get_data()).encode('ascii', 'ignore')
				overview=overview.replace('\n',' ')
				self.overviewlist+=[overview]
				self.comp_list+=[unicodedata.normalize('NFKD', 
            		cp.get('name')).encode('ascii', 'ignore')]
				self.overview_tag[cp.get('tag_list')]=overview

		self.num_doc = len(self.overviewlist)

	def Vectorize(self):
		self.vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
						 stop_words = 'english')
		self.vectorized = self.vectorizer.fit_transform(self.overviewlist)
		#num_samples, num_features = vectorized.shape
		

	def NMF(self):
		
		n_topics = 100
		n_top_words = 20
		self.nmf = decomposition.NMF(n_components=n_topics).fit(self.vectorized)

		self.feature_names = self.vectorizer.get_feature_names()

		self.topic=[]
		for topic_idx, topic in enumerate(self.nmf.components_):
			#print "Topic #%d:" % topic_idx
			self.topic+=[ " ".join([self.feature_names[i] \
				           for i in topic.argsort()[:-n_top_words - 1:-1]])	]

		self.weight_mat=self.nmf.fit_transform(self.vectorized)

		self.topic_id=[]
		for i in xrange(len(self.weight_mat)):
			idx=numpy.argsort(self.weight_mat[i])
			self.topic_id+=[idx]
	
	def clustering(self,num_clusters=20):
		km = KMeans(n_clusters = num_clusters, init = 'k-means++', n_init = 1,
            verbose=1)
		clustered = km.fit(self.weight_mat)
		self.predict = km.predict(self.weight_mat)

	def create_csv(self):
		Comp_clust = zip(self.comp_list, self.predict)
		Comp_df = DataFrame(data = Comp_clust, columns = ['Company','Cluster'])
		Comp_df.to_csv('CompanyCluster.csv', index = False)

	