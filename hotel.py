import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS 

import warnings 
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# HOW MANY POSITIVE, NEGATIVE & NEUTRAL COMMENTS

data = pd.read_csv("/Users/emmaalgotsson/Documents/Kurser/TNM108-Maskin/Projekt/Datafiniti_Hotel_Reviews_Jun19.csv")

# data.shape # 10000, 26
# data.columns
'''
Index(['id', 'dateAdded', 'dateUpdated', 'address', 'categories',
       'primaryCategories', 'city', 'country', 'keys', 'latitude', 'longitude',
       'name', 'postalCode', 'province', 'reviews.date', 'reviews.dateAdded',
       'reviews.dateSeen', 'reviews.rating', 'reviews.sourceURLs',
       'reviews.text', 'reviews.title', 'reviews.userCity',
       'reviews.userProvince', 'reviews.username', 'sourceURLs', 'websites'],
      dtype='object')
'''

#not important columns
columns = ['id', 'dateAdded', 'dateUpdated', 'categories', 'primaryCategories',
            'country', 'keys', 'reviews.date', 'reviews.dateAdded', 'reviews.dateSeen',
            'reviews.sourceURLs', 'reviews.title', 'reviews.userCity', 'reviews.userProvince',
            'reviews.username', 'sourceURLs', 'websites']

# drop the unimportant columns
df = pd.DataFrame(data.drop(columns, axis=1, inplace=False))

#number of every rating, show diagram
df['reviews.rating'].value_counts().plot(kind='bar')
# print(data.describe())
# data.hist()
#plt.show()

# ---- Pre-process reviews text before calculating sentiment score -------

# change reviews type to string
df['reviews.text'] = df['reviews.text'].astype(str)

# before lowercasing
df['reviews.text'][3] #test review
#Not cheap but excellent location. Price is somewhat standard for not hacing reservations. 
#But room was nice and clean. They offer good continental breakfast which is a plus and compensates. 
#Front desk service and personnel where excellent. It is Carmel, no A/C in rooms but they have a fan for air circulation.

# Lowercase all reviews
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['reviews.text'][3] #test review
#not cheap but excellent location. price is somewhat standard for not hacing reservations. 
#but room was nice and clean. they offer good continental breakfast which is a plus and compensates. 
#front desk service and personnel where excellent. it is carmel, no a/c in rooms but they have a fan for air circulation.

# remove punctuation
df['reviews.text'] = df['reviews.text'].str.replace('[^\w\s]','')
df['reviews.text'][3] #test review
#not cheap but excellent location price is somewhat standard for not hacing reservations 
#but room was nice and clean they offer good continental breakfast which is a plus and compensates 
#front desk service and personnel where excellent it is carmel no ac in rooms but they have a fan for air circulation

# stopwords
stop = stopwords.words('english')
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['reviews.text'][3] #test review
#cheap excellent location price somewhat standard hacing reservations room nice clean offer good continental 
#breakfast plus compensates front desk service personnel excellent carmel ac rooms fan air circulation

# stemming, reduces words with the same roots
st = PorterStemmer()
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df['reviews.text'][3] #test review
#cheap excel locat price somewhat standard hace reserv room nice clean offer good continent breakfast plu compens 
#front desk servic personnel excel carmel ac room fan air circul

most = pd.Series(' '.join(df['reviews.text']).split()).value_counts()[:10]

most = list(most.index)
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x for x in x.split() if x not in most))
df['reviews.text'].head()

# --- sentiment score, calculate score for the whole dataset
# def senti(x):
#     return TextBlob(x).sentiment

# df['senti_score'] = df['reviews.text'].apply(senti)
# df.senti_score.head()

# ------------- separate polarity and subjectivity
polarity = []
subjectivity = []

for i in df['reviews.text'].values:
  try:
    analysis = TextBlob(i)
    polarity.append(analysis.sentiment.polarity)
    subjectivity.append(analysis.sentiment.subjectivity)
  except:
    polarity.append(0)
    subjectivity.append(0)

df['polarity'] = polarity
df['subjectivity'] = subjectivity

#negative comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity<0].head(10)
#positive comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity>0].head(10)
#neutral comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity==0].head(10)
#highly subjective reviews
df[['name','reviews.text','polarity','subjectivity']][df.subjectivity>0.8].head(10)
#highly positive comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity>0.8].head(10)
# highly negative comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity<-0.4].head(10)

#converting polarity values from continuous to categorical
df['polarity'][df.polarity==0]=0 #neutral -> 486
df['polarity'][df.polarity > 0] = 1 #positive -> 8712
df['polarity'][df.polarity < 0] = -1 #negative -> 802

#number of negative, positive and neutral comments
df.polarity.value_counts().plot.bar()
print(df.polarity.value_counts())

# -----------------------------------------------------------

#location of hotels
# data.plot(x='latitude', y='longitude', style='o')
# plt.title('Latitude - Longitude')
# plt.xlabel('latitude')
# plt.ylabel('longitude')
# plt.show()

