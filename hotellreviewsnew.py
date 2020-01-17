import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 300)

import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

#print(TextBlob("The movie is good").sentiment)


data = pd.read_csv("C:/Users/Erica/Documents/TNM108/projekt/Datafiniti_Hotel_Reviews_Jun19.csv")
from wordcloud import WordCloud

data.head()

def wc(data,bgcolor,title):
    plt.figure(figsize = (50, 50))
    wc = WordCloud(background_color = bgcolor, max_words = 2000, random_state=42, max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

print("There are {} observations and {} features in this dataset. \n".format(data.shape[0], data.shape[1]))
