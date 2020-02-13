import string
import collections
 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import numpy as np #from numpy package
import sklearn.cluster  # from sklearn package
# import distance #from distance package
import fnmatch
import glob,os
from sklearn.decomposition import LatentDirichletAllocation as LDA
from tika import parser

# import site; 
# print site.getsitepackages()
def filemaker(name_of_file):
	with open(name_of_file) as f:
		doc = slate.PDF(f)	
	data  = np.asarray(doc)
	return data

matches = []
os.chdir("./")
for file in glob.glob("*.pdf"):
    matches.append(file)

out=[]
for f_names in matches:
	try:
		raw = parser.from_file(f_names)
		out.append(raw['content'])
	except:
		print("failed!")
		pass  

def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
 
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens
 
 
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
 
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(texts[idx])
 
    return clustering	

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

clusters=cluster_texts(out,15)    

number_topics = 3
number_words = 5

for key in clusters:
	print("Cluster "+str(key))
	count_vectorizer = CountVectorizer(stop_words='english')
	count_data = count_vectorizer.fit_transform(clusters[key])
	lda = LDA(n_components=number_topics, n_jobs=-5)
	lda.fit(count_data)
	print("Topics found via LDA:")
	print_topics(lda, count_vectorizer, number_words)
