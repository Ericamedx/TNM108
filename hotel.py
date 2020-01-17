import pandas as pd
import numpy as np 
pd.set_option('display.max_colwidth', 300)

import matplotlib.pyplot as plt 
import seaborn as sns

from textblob import TextBlob
import warnings 
warnings.filterwarnings("ignore")

from nlk.tokenize import word_tokenize, sent_tokenize

# load data 
data = pd.read_csv("Datafiniti_Hotel_Reviews_Jun19.csv")
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
comm[['name','reviews.text','polarity','subjectivity']][comm.polarity<0].head(10)

#positive comments
comm[['name','reviews.text','polarity','subjectivity']][comm.polarity>0].head(10)

#neutral comments
comm[['name','reviews.text','polarity','subjectivity']][comm.polarity==0].head(10)

#highly subjective reviews
comm[['name','reviews.text','polarity','subjectivity']][comm.subjectivity>0.8].head(10)

#highly positive
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
comm['polarity'][comm.polarity==0]=0 #16
comm['polarity'][comm.polarity > 0]= 1 #4748
comm['polarity'][comm.polarity < 0]= -1 #236

#antalet positiva negativa och neutrala commentarer
comm.polarity.value_counts().plot.bar()
print(comm.polarity.value_counts())
#plt.show()



