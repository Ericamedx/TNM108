# ------------------------------------------------------------------------
# Code written by: Emma Algotsson, emmal084, and Erica Ahlqvist, eriah648,
#
# TNM108: SENTIMENT ANALYSIS OF HOTEL REVIEWS
# -------------------------------------------------------------------------

#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from wordcloud import WordCloud

# ignore warnings
import warnings 
warnings.filterwarnings("ignore")


# HOW MANY POSITIVE, NEGATIVE & NEUTRAL COMMENTS

# load data
data = pd.read_csv("Datafiniti_Hotel_Reviews_Jun19.csv")

# look at the dataset
# print(data.shape) # 10000, 26
# print(data.columns) #All columns in the dataset
'''
Index(['id', 'dateAdded', 'dateUpdated', 'address', 'categories',
       'primaryCategories', 'city', 'country', 'keys', 'latitude', 'longitude',
       'name', 'postalCode', 'province', 'reviews.date', 'reviews.dateAdded',
       'reviews.dateSeen', 'reviews.rating', 'reviews.sourceURLs',
       'reviews.text', 'reviews.title', 'reviews.userCity',
       'reviews.userProvince', 'reviews.username', 'sourceURLs', 'websites'],
      dtype='object')
'''

# columns that will not be necessary in this project
columns = ['id', 'dateAdded', 'dateUpdated', 'address', 'categories', 'primaryCategories',
            'country', 'keys', 'latitude', 'longitude', 'postalCode', 'province', 
            'reviews.date', 'reviews.dateAdded', 'reviews.dateSeen',
            'reviews.sourceURLs', 'reviews.title', 'reviews.userCity', 'reviews.userProvince',
            'reviews.username', 'sourceURLs', 'websites']

# drop columns
df = pd.DataFrame(data.drop(columns, axis=1, inplace=False))

# look at reviews length
df['reviews length'] = df['reviews.text'].apply(len)
sns.set_style('white')
g = sns.FacetGrid(df,col='reviews.rating')
g.map(plt.hist,'reviews length')

#number of every rating, show diagram

x=df['reviews.rating'].value_counts()
x=x.sort_index()
plt.figure(figsize=(10,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Star Rating Distribution")
plt.ylabel('count')
plt.xlabel('Star Ratings')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show();

# ---- Pre-process reviews text before calculating sentiment score -------

# change reviews type to string
df['reviews.text'] = df['reviews.text'].astype(str)

# before lowercasing
df['reviews.text'][3] #test a review
#Not cheap but excellent location. Price is somewhat standard for not hacing reservations. 
#But room was nice and clean. They offer good continental breakfast which is a plus and compensates. 
#Front desk service and personnel where excellent. 
#It is Carmel, no A/C in rooms but they have a fan for air circulation.


# Lowercase all reviews
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['reviews.text'][3] #test a review
#not cheap but excellent location. price is somewhat standard for not hacing reservations. 
#but room was nice and clean. they offer good continental breakfast which is a plus and compensates. 
#front desk service and personnel where excellent. 
#it is carmel, no a/c in rooms but they have a fan for air circulation.

# remove punctuation
df['reviews.text'] = df['reviews.text'].str.replace('[^\w\s]','')
df['reviews.text'][3] #test a review
#not cheap but excellent location price is somewhat standard for not hacing reservations 
#but room was nice and clean they offer good continental breakfast which is a plus and compensates 
#front desk service and personnel where excellent 
#it is carmel no ac in rooms but they have a fan for air circulation

# remove stop words
stop = stopwords.words('english')
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['reviews.text'][3] #test a review
#cheap excellent location price somewhat standard hacing reservations room nice clean offer good continental 
#breakfast plus compensates front desk service personnel excellent carmel ac rooms fan air circulation

# stemming, reduces words with the same roots
st = PorterStemmer()
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df['reviews.text'][3] #test review
#cheap excel locat price somewhat standard hace reserv room nice clean offer good continent breakfast plu compens 
#front desk servic personnel excel carmel ac room fan air circul

#join all
most = pd.Series(' '.join(df['reviews.text']).split()).value_counts()[:10]

most = list(most.index)
df['reviews.text'] = df['reviews.text'].apply(lambda x: " ".join(x for x in x.split() if x not in most))
df['reviews.text'].head()

# -------------------------- Sentiment Analysis ------------------------------------

# separate polarity and subjectivity parameters
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

# look at the distribution of polarity from reviews.text 
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df.polarity, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Histogram of polarity')
plt.show();

# look at some of the comments

#negative comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity<0].head(10)
#positive comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity>0].head(10)
#neutral comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity==0].head(10)

#highly subjective reviews
df[['name','reviews.text','polarity','subjectivity']][df.subjectivity>0.8].head(10)
#highly objective reviews 
df[['name','reviews.text','polarity','subjectivity']][df.subjectivity<0.2].head(10)

#highly positive comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity>0.8].head(10)

#print common words with high polarity score, wordcloud
def wc(data,bgcolor,title):
    plt.figure(figsize = (20,20))
    wc = WordCloud(background_color = bgcolor, max_words = 200, random_state=42, max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
wc(df['reviews.text'][df.polarity>0.8],'black', 'Common Words')
plt.show()

# highly negative comments
df[['name','reviews.text','polarity','subjectivity']][df.polarity<-0.4].head(10)

#converting polarity values from continuous to categorical
df['polarity'][df.polarity == 0]=0 #neutral -> 486
df['polarity'][df.polarity > 0] = 1 #positive -> 8712
df['polarity'][df.polarity < 0] = -1 #negative -> 802

# show categorical polarity distribution
x=df.polarity.value_counts()
x=x.sort_index()
plt.figure(figsize=(10,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Polarity Distribution")
plt.ylabel('count')
plt.xlabel('Polarity value')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show();

# ------------------ Lets take look at the subjectivity property -------------------------

#converting polarity values from continuous to categorical
df['subjectivity'][df.subjectivity > 0.5] = 1 #subjective
df['subjectivity'][df.subjectivity == 0.5] = 0.5  #neutral
df['subjectivity'][df.subjectivity < 0.5] = 0 #objective

# print categorical subjectivity distribution
x=df.subjectivity.value_counts()
x=x.sort_index()
plt.figure(figsize=(10,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Subjectivity Distribution")
plt.ylabel('count')
plt.xlabel('Subjectivity value')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show();

