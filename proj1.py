import glob
import re
import string
import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.corpus import words
from yellowbrick.text import TSNEVisualizer
from yellowbrick.text import FreqDistVisualizer

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

char_count=[]
documents = []
for filename in glob.glob('polarity_html/movie/*.html'):
    with open(filename, 'rb') as f:
        raw = f.read()
        char_count.append(len(raw))
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


#going to see most common words and remove based on usefulness, remove document/domain specific stopwords
#using built in english stopwords
#removing domain specific stop words based on the previous list
domain_specific_stop_words = ["movie", "review","reviewed", "copyright", "film", "story", "plot", "director", "characters", "character", "film", "scene", "scenes"]
stop_words = text.ENGLISH_STOP_WORDS.union(domain_specific_stop_words)

#min df ignore term that only occur in 1 percent of documents, max df ifnore therms that occur in more than half of the documents
#removed about 1/4 of the words
tfidf_vect = TfidfVectorizer(stop_words= stop_words, decode_error='ignore', min_df=0.01, max_df=0.70)
tfidf_bag_words = tfidf_vect.fit_transform(documents)
tf_df = pd.DataFrame(data=tfidf_bag_words.toarray(),columns=tfidf_vect.get_feature_names())

df.sum().sort_values()[-10:]
tf_df.sum().sort_values()[-10:]



#     ____  ____  _____(_) /_(_)   _____     ____ _____  ____/ /  ____  ___  ____ _____ _/ /_(_)   _____
#    / __ \/ __ \/ ___/ / __/ / | / / _ \   / __ `/ __ \/ __  /  / __ \/ _ \/ __ `/ __ `/ __/ / | / / _ \
#   / /_/ / /_/ (__  ) / /_/ /| |/ /  __/  / /_/ / / / / /_/ /  / / / /  __/ /_/ / /_/ / /_/ /| |/ /  __/
#  / .___/\____/____/_/\__/_/ |___/\___/   \__,_/_/ /_/\__,_/  /_/ /_/\___/\__, /\__,_/\__/_/ |___/\___/
# /_/                                                                     /____/


#using postive sentiment vocabulary
positive_vocab_file = open("positive_sentiment_indicators.txt","r")
positive_voc = positive_vocab_file.read().split('\n')

pos_count_vect = CountVectorizer(stop_words= stop_words,
                             decode_error='ignore',
                             vocabulary=positive_voc
                                )

pos_bag_words = pos_count_vect.fit_transform(documents)
pos_bag_words.shape
pos_df = pd.DataFrame(data=pos_bag_words.toarray(), columns=positive_voc)


#using negative sentiment vocabulary
negative_vocab_file = open("negative_sentiment_indicators.txt","r")
negative_voc = negative_vocab_file.read().split('\n')

neg_count_vect = CountVectorizer(stop_words= stop_words,
                             decode_error='ignore',
                             vocabulary=negative_voc
                                )

neg_bag_words = neg_count_vect.fit_transform(documents)
neg_bag_words.shape
neg_df = pd.DataFrame(data=neg_bag_words.toarray(), columns=negative_voc)



#negative words in first document
neg_count_vect.inverse_transform(neg_bag_words[0])
#postive words in first document
pos_count_vect.inverse_transform(pos_bag_words[0])

neg_df.sum().sort_values()[-10:]
pos_df.sum().sort_values()[-10:]




#
#    ____  _
#    / __ )(_)___ __________ _____ ___  _____
#   / __  / / __ `/ ___/ __ `/ __ `__ \/ ___/
#  / /_/ / / /_/ / /  / /_/ / / / / / (__  )
# /_____/_/\__, /_/   \__,_/_/ /_/ /_/____/
#         /____/


#bigram count
bigram_count_vect = CountVectorizer(stop_words= stop_words,
                             decode_error='ignore',
                             ngram_range=(2, 2)
                                ) # an object capable of counting words in a document!
num_limit = int(len(documents)/100)
bigram_bag_words = bigram_count_vect.fit_transform(documents[:num_limit])
print(bigram_bag_words.shape) # this is a sparse matrix
print('=========')
bigram_count_vect.inverse_transform(bigram_bag_words[0])
bi_count_df = pd.DataFrame(data=bigram_bag_words.toarray(),columns=bigram_count_vect.get_feature_names())


# bigram tdidf
bi_tfidf_vect = TfidfVectorizer(stop_words= stop_words, decode_error='ignore', ngram_range=(2, 2), min_df=0.01, max_df=0.70)
num_limit = int(len(documents)/100)
bi_tfidf_mat = bi_tfidf_vect.fit_transform(documents[:num_limit])
bi_td_df = pd.DataFrame(data=bi_tfidf_mat.toarray(), columns=bi_tfidf_vect.get_feature_names())



#bigram with adverb
adverb_file = open("adverbs.txt","r")
adverbs_voc = adverb_file.read().split('\n')

adv_neg = list(map( lambda x: x[0]+ " " + x[1], itertools.product(adverbs_voc, negative_voc) ))
adv_pos = list(map( lambda x: x[0]+ " " + x[1], itertools.product(adverbs_voc, positive_voc) ))

adv_with_adj = adv_pos + adv_neg
adv_with_adj = list(set(adv_with_adj))

adv_bi_count_vect = CountVectorizer(stop_words= stop_words,
                             decode_error='ignore',
                             ngram_range=(2, 2),
                             vocabulary=adv_with_adj
                                )
num_limit = int(len(documents)/10)
adv_bi_bag_words = adv_bi_count_vect.fit_transform(documents[:num_limit])



documents[10]
adv_bi_count_vect.inverse_transform(adv_bi_bag_words[10])
adv_bi_df = pd.DataFrame(data=adv_bi_bag_words.toarray(),columns=adv_bi_count_vect.get_feature_names())



bi_count_df.sum().sort_values()[-10:]
bi_td_df.sum().sort_values()[-10:]
adv_bi_df.sum().sort_values()[-10:]


#                               __      __        ____
#    ____  ___ _      __   ____/ /___ _/ /_____ _/ __/________ _____ ___  ___
#   / __ \/ _ \ | /| / /  / __  / __ `/ __/ __ `/ /_/ ___/ __ `/ __ `__ \/ _ \
#  / / / /  __/ |/ |/ /  / /_/ / /_/ / /_/ /_/ / __/ /  / /_/ / / / / / /  __/
# /_/ /_/\___/|__/|__/   \__,_/\__,_/\__/\__,_/_/ /_/   \__,_/_/ /_/ /_/\___/



#Statistical dataframes by document. columns are pos, neg, vocab size, word number, greatest controversial number,


data_stats = pd.DataFrame()
length = neg_bag_words.shape[0]
data_stats['positive_word_count'] = [ pos_count_vect.inverse_transform(pos_bag_words[doc])[0].size for doc in range(length)]
data_stats['negative_word_count'] = [ neg_count_vect.inverse_transform(neg_bag_words[doc])[0].size for doc in range(length)]
data_stats['total_char_count'] = char_count
# data_stats['total_vocab_count'] = [ count_vect.inverse_transform(bag_words[doc])[0].size for doc in range(length)]

data_stats["sentiment_score"] = data_stats.apply(lambda row: row.positive_word_count - row.negative_word_count, axis=1)
data_stats["sentiment_occurences"] = data_stats.apply(lambda row: row.positive_word_count + row.negative_word_count, axis=1)

def sentiment_classifier(row):

    if row.sentiment_occurences == 0:
        return 0

    score = row.sentiment_score/row.sentiment_occurences

    if score > 0.333:
        #the good is double the bad
        return 3
    elif score > 0.2:
        #the good is 50% more the bad
        return 2
    elif score > 0.111:
        #the good is 25% more the bad
        return 1
    elif score < -0.333:
        #the bad is double the good
        return -3
    elif score < -0.2:
        #the bad is 50% more the good
        return -2
    elif score < -0.111:
        #the bad is 25% more the good
        return -1
    else:
        return 0

data_stats["sentiment_class"] = data_stats.apply(sentiment_classifier, axis=1)
data_stats

df_grouped_sentiments = data_stats.groupby(by='sentiment_class')
for val,grp in df_grouped_sentiments:
    print('There were',len(grp),'reviews sentimentally rated',val)

data_stats.describe


#
#  _    ___                  ___             __  _
# | |  / (_)______  ______ _/ (_)___  ____ _/ /_(_)___  ____
# | | / / / ___/ / / / __ `/ / /_  / / __ `/ __/ / __ \/ __ \
# | |/ / (__  ) /_/ / /_/ / / / / /_/ /_/ / /_/ / /_/ / / / /
# |___/_/____/\__,_/\__,_/_/_/ /___/\__,_/\__/_/\____/_/ /_/
#
#




sns.set(color_codes=True)

import warnings
warnings.filterwarnings('ignore')
sns.distplot(data_stats.sentiment_score);
sns.distplot(data_stats.positive_word_count);
sns.distplot(data_stats.negative_word_count);
sns.distplot(data_stats.sentiment_occurences);


sns.violinplot(data=data_stats, x="sentiment_score", y="sentiment_occurences", hue="sentiment_class", split=True, inner="box")



#interesting note that the sentiment score is more related to the negative occurences than positive.
drop= data_stats.drop(columns= ["sentiment_class"])
# plot the correlation matrix using seaborn
cmap = sns.set(style="darkgrid") # one of the many styles to plot using

f, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(drop.corr(), cmap=cmap, annot=True)

f.tight_layout()




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
