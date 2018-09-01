import glob
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.corpus import words
from yellowbrick.text import TSNEVisualizer

#vocabulary from:
#http://ptrckprry.com/course/ssd/data/negative-words.txt
#http://ptrckprry.com/course/ssd/data/positive-words.txt




#read in document raw and preprocess
#removing punctutation and return characters as as well as html no_tags
#using fast c level code such as translate and regex to improve speed
#read in as byte code to drastically improve speed
#over 30000 reviews documents in html text format
def preprocess(text):
    text= re.sub(b"<.*?>", b" ", text)#no_tags
    text= re.sub(b"\n", b" ", text)#no_new_lines
    text= re.sub(b"\r", b" ", text)#no_returns
    #lowered with no punctuation
    text= text.translate(None, b'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~').lower()
    #removing the footer for all the reviews
    text= text[:-579]
    return text


documents = []
for filename in glob.glob('polarity_html/movie/*.html'):
    with open(filename, 'rb') as f:
        raw = f.read()
        cleaned = preprocess(raw)
        documents.append(cleaned)


print(len(documents))

#code to see raw vesion
example_raw = open("polarity_html/movie/0020.html")
example_raw.read()

#code to view cleaned version
documents[0]




#          __                                         __                        __        __
#    _____/ /_____  ____     _      ______  _________/ /____   ____ _____  ____/ /  _____/ /__  ____ _____  __  ______
#   / ___/ __/ __ \/ __ \   | | /| / / __ \/ ___/ __  / ___/  / __ `/ __ \/ __  /  / ___/ / _ \/ __ `/ __ \/ / / / __ \
#  (__  ) /_/ /_/ / /_/ /   | |/ |/ / /_/ / /  / /_/ (__  )  / /_/ / / / / /_/ /  / /__/ /  __/ /_/ / / / / /_/ / /_/ /
# /____/\__/\____/ .___/    |__/|__/\____/_/   \__,_/____/   \__,_/_/ /_/\__,_/   \___/_/\___/\__,_/_/ /_/\__,_/ .___/
#               /_/                                                                                           /_/







#no settings other than english stop words
count_vect = CountVectorizer(stop_words= 'english',
                             decode_error='ignore'
                                ) # an object capable of counting words in a document!

bag_words = count_vect.fit_transform(documents)
print(bag_words)
print(bag_words.shape)
print(count_vect.inverse_transform(bag_words[0]))

#find words with most count, see possible stop words that have not been coverted
df = pd.DataFrame(data=bag_words.toarray(), columns= count_vect.get_feature_names())
df.sum().sort_values()[-50:]

#going to see most common words and remove based on usefulness, remove document/domain specific stopwords
#using built in english stopwords
#removing domain specific stop words based on the previous list
domain_specific_stop_words = ["movie", "review","reviewed", "copyright", "film", "story", "plot", "director", "characters", "character", "film", "scene", "scenes"]
stop_words = text.ENGLISH_STOP_WORDS.union(domain_specific_stop_words)

#min df ignore term that only occur in 1 percent of documents, max df ifnore therms that occur in more than half of the documents
#removed about 1/4 of the words
tfidf_vect = TfidfVectorizer(stop_words= stop_words, decode_error='ignore', min_df=0.01, max_df=0.70)
tfidf_mat = tfidf_vect.fit_transform(documents)
tfidf_mat

# now let's create a pandas API out of this
pd.options.display.max_columns = 999
df = pd.DataFrame(data=tfidf_mat.toarray(),columns=tfidf_vect.get_feature_names())
# print out 10 most common words in our data
df.sum().sort_values()[-50:]







#     ____  ____  _____(_) /_(_)   _____     ____ _____  ____/ /  ____  ___  ____ _____ _/ /_(_)   _____
#    / __ \/ __ \/ ___/ / __/ / | / / _ \   / __ `/ __ \/ __  /  / __ \/ _ \/ __ `/ __ `/ __/ / | / / _ \
#   / /_/ / /_/ (__  ) / /_/ /| |/ /  __/  / /_/ / / / / /_/ /  / / / /  __/ /_/ / /_/ / /_/ /| |/ /  __/
#  / .___/\____/____/_/\__/_/ |___/\___/   \__,_/_/ /_/\__,_/  /_/ /_/\___/\__, /\__,_/\__/_/ |___/\___/
# /_/                                                                     /____/







#using postive sentiment vocabulary
positive_vocab_file = open("positive_sentiment_indicators.txt","r")
positive_voc = positive_vocab_file.read().split('\n')

count_vect = CountVectorizer(stop_words= stop_words,
                             decode_error='ignore',
                             vocabulary=positive_voc
                                ) # an object capable of counting words in a document!

bag_words = count_vect.fit_transform(documents)
bag_words.shape
# now let's create a pandas API out of this
pd.options.display.max_columns = 999
cols=count_vect.get_feature_names()
df = pd.DataFrame(data=bag_words.toarray(), columns=positive_voc)
# print out 10 most common words in our data
df.sum().sort_values()[-50:]


#postive words in first document
count_vect.inverse_transform(bag_words[0])




#using negative sentiment vocabulary
negative_vocab_file = open("negative_sentiment_indicators.txt","r")
negative_voc = negative_vocab_file.read().split('\n')

count_vect = CountVectorizer(stop_words= stop_words,
                             decode_error='ignore',
                             vocabulary=negative_voc
                                ) # an object capable of counting words in a document!

bag_words = count_vect.fit_transform(documents)
bag_words.shape
# now let's create a pandas API out of this
pd.options.display.max_columns = 999
df = pd.DataFrame(data=bag_words.toarray(), columns=negative_voc)
# print out 10 most common words in our data
df.sum().sort_values()[-10:]


#negative words in first document
count_vect.inverse_transform(bag_words[0])

preprocess(documents[0])


#
#    ____  _
#    / __ )(_)___ __________ _____ ___  _____
#   / __  / / __ `/ ___/ __ `/ __ `__ \/ ___/
#  / /_/ / / /_/ / /  / /_/ / / / / / (__  )
# /_____/_/\__, /_/   \__,_/_/ /_/ /_/____/
#         /____/


#bigram
count_vect = CountVectorizer(stop_words= stop_words,
                             decode_error='ignore',
                             ngram_range=(2, 2)
                                ) # an object capable of counting words in a document!
num_limit = int(len(documents)/100)
bag_words = count_vect.fit_transform(documents[:num_limit])
print(bag_words.shape) # this is a sparse matrix
print('=========')
count_vect.inverse_transform(bag_words[0])


# now let's create a pandas API out of this
pd.options.display.max_columns = 999
df = pd.DataFrame(data=bag_words.toarray(),columns=count_vect.get_feature_names())


# bigram tdidf
df.sum().sort_values()[-100:]

tfidf_vect = TfidfVectorizer(stop_words= stop_words, decode_error='ignore', ngram_range=(2, 2), min_df=0.01, max_df=0.70)
tfidf_mat = tfidf_vect.fit_transform(documents)
num_limit = int(len(documents)/100)
bag_words = count_vect.fit_transform(documents[:num_limit])
print(bag_words.shape) # this is a sparse matrix
print('=========')
count_vect.inverse_transform(bag_words[0])

#bigram with adverb



# now let's create a pandas API out of this
pd.options.display.max_columns = 999
df = pd.DataFrame(data=bag_words.toarray(),columns=count_vect.get_feature_names())
df.sum().sort_values()[-100:]



#
#  _    ___                  ___             __  _
# | |  / (_)______  ______ _/ (_)___  ____ _/ /_(_)___  ____
# | | / / / ___/ / / / __ `/ / /_  / / __ `/ __/ / __ \/ __ \
# | |/ / (__  ) /_/ / /_/ / / / / /_/ /_/ / /_/ / /_/ / / / /
# |___/_/____/\__,_/\__,_/_/_/ /___/\__,_/\__/_/\____/_/ /_/
#
#

tfidf = TfidfVectorizer(decode_error='ignore')
docs = tfidf.fit_transform(documents)

from sklearn.cluster import KMeans

clusters = KMeans(n_clusters=2)
clusters.fit(docs)

tsne = TSNEVisualizer()
tsne.fit(docs, ["c{}".format(c) for c in clusters.labels_])
tsne.poof()

# pd.options.display.max_columns = 999
# df = pd.DataFrame(data=bag_words.toarray(),columns=count_vect.get_feature_names())
# df
# df.max().sort_values()[-10:]
tfidf = TfidfVectorizer(decode_error='ignore')
docs = tfidf.fit_transform(documents)


tsne = TSNEVisualizer(labels=["documents"])


tsne.fit(docs)
tsne.poof()
