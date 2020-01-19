import pandas as pd
import numpy as np 
pd.set_option('display.max_colwidth', 300)
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import warnings 
warnings.filterwarnings("ignore")
from wordcloud import WordCloud

# load data 
data = pd.read_csv("/Users/emmaalgotsson/Documents/Kurser/TNM108-Maskin/Projekt/Datafiniti_Hotel_Reviews_Jun19.csv")

#check dataset ----------------
# print(data.shape)
# print(data.describe())
# print(data.describe(include=['O']))
# print(data.keys())

# -------------------------------------------------------------------
# HOW MANY POSITIVE, NEGATIVE & NEUTRAL COMMENTS

import matplotlib.pyplot as plt 
import seaborn as sns

from textblob import TextBlob
import warnings 
warnings.filterwarnings("ignore")

from nltk.tokenize import word_tokenize, sent_tokenize

# load data 
data = pd.read_csv("/Users/emmaalgotsson/Documents/Kurser/TNM108-Maskin/Projekt/Datafiniti_Hotel_Reviews_Jun19.csv")
from wordcloud import WordCloud

def wc(data,bgcolor,title):
    plt.figure(figsize = (50, 50))
    wc = WordCloud(background_color = bgcolor, max_words = 2000, random_state=42, max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
    
comm = data.sample(5000)
comm.shape

polarity = []
subjectivity = []

for i in comm['reviews.text'].values:
    try:
        analysis = TextBlob(i)
        polarity.append(analysis.sentiment.polarity)
        subjectivity.append(analysis.sentiment.subjectivity)

    except:
        polarity.append(0)
        subjectivity.append(0)

comm['polarity']= polarity
comm['subjectivity']=subjectivity

#negative comments

neg_rev = comm[['name','reviews.text','polarity','subjectivity']][comm.polarity<0].head(10)

comm[['name','reviews.text','polarity','subjectivity']][comm.polarity<0].head(10)

#positive comments
comm[['name','reviews.text','polarity','subjectivity']][comm.polarity>0].head(10)

#neutral comments
comm[['name','reviews.text','polarity','subjectivity']][comm.polarity==0].head(10)

#highly subjective reviews
comm[['name','reviews.text','polarity','subjectivity']][comm.subjectivity>0.8].head(10)

#highly positive
pos_rev = comm[['name','reviews.text','polarity','subjectivity']][comm.polarity>0.8].head(10)

comm[['name','reviews.text','polarity','subjectivity']][comm.polarity>0.8].head(10)


#wc(comm['reviews.text'][comm.polarity>0.8],'black', 'Common Words')

# fig = plt.figure(1, figsize = (20,20))
# plt.axis('off')
# plt.show()

#higly negative
comm[['name','reviews.text','polarity','subjectivity']][comm.polarity<-0.4].head(10)

# wc(comm['reviews.text'][comm.polarity<-0.4],'black', 'Common Words')

# fig = plt.figure(1, figsize = (20,20))
# plt.axis('off')
# plt.show()

#comm.polarity.hist(bins=50)
#plt.show()
#comm.subjectivity.hist(bins=50)
#plt.show()

# converting polarity values from continuous to categorical
comm['polarity'][comm.polarity==0]=0 #19
comm['polarity'][comm.polarity > 0]= 1 #4743
comm['polarity'][comm.polarity < 0]= -1 #238

#antalet positiva negativa och neutrala kommentarer
comm.polarity.value_counts().plot.bar()
comm.polarity.value_counts()
#plt.show()


from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer()

# Stopwords, numbers and punctuation to remove
remove_punct_and_digits = dict([(ord(punct), ' ') for punct in string.punctuation + string.digits])
stopWords = set(stopwords.words('english'))

def word_cleaner(text):
    cleaned_word = text.lower().translate(remove_punct_and_digits)
    words = word_tokenize(cleaned_word)
    words = [toktok.tokenize(sent) for sent in sent_tokenize(cleaned_word)]
    wordsFiltered = []
    if not words:
        pass
    else:
        for w in words[0]:
            if w not in stopWords:
                wordsFiltered.append(w)
                end=time.time()
    return wordsFiltered


#antalet positiva negativa och neutrala commentarer
comm.polarity.value_counts().plot.bar()
print(comm.polarity.value_counts())
#plt.show()


