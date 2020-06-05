# imports pandas, numpy and matplotlib modules
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib inline

# import plotly modules
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings('ignore')

## load csv data
nRowsRead = None

hotrev = pd.read_csv("/Users/emmaalgotsson/Documents/Kurser/TNM108-Maskin/Projekt/Datafiniti_Hotel_Reviews_Jun19.csv", delimiter=',', nrows = nRowsRead)

ndarcity = (np.unique(hotrev['city']))
ndarhotel = (np.unique(hotrev['name']))

#Finding total number of Unique Cities and Hotels
lencity = len(ndarcity)
lenhotel = len(ndarhotel)

print('Number of distinct city = ',lencity)
print('Number of distinct hotels = ',lenhotel,'\n')

# -----------------------------------------------------------------

#City number of hotels
city_numof_hotels = pd.DataFrame(columns=['City', 'NumofHotels'])

for i in range(lencity):
    city_numof_hotels.loc[i] = (ndarcity[i],len(np.unique(hotrev['name'][hotrev['city']==ndarcity[i]])))

city_numof_hotels.to_csv('Derived_City_Numof_Hotels.csv')
#print(city_numof_hotels)
print('City with Maximum hotels:')
print((city_numof_hotels['City'][city_numof_hotels['NumofHotels'] == max(city_numof_hotels['NumofHotels'])]),max(city_numof_hotels['NumofHotels']))

# ------------------------------------------------------------------

#Hotel Avg Review
hotels_list = pd.DataFrame(columns=['Hotel_Name','Avg_Review', 'Percentage'])

for i in range(1311):
    hotels_list.loc[i] = (ndarhotel[i],(np.mean(hotrev['reviews.rating'][hotrev['name'] == ndarhotel[i]])),((np.mean(hotrev['reviews.rating'][hotrev['name'] == ndarhotel[i]]))/5))

hotels_list.to_csv('Derived_Hotel_List_Review.csv')
#print(hotels_list)
print('Hotel with Maximum Average Review')
print((hotels_list['Hotel_Name'][hotels_list['Avg_Review'] == max(hotels_list['Avg_Review'])]),max(hotels_list['Avg_Review']))
#print (np.mean(hotrev['reviews.rating'][hotrev['name'] == 'Agate Beach Motel']))

l = list(range(len(hotels_list['Hotel_Name'])))
plt.xticks(l,hotels_list['Hotel_Name'],rotation = 'vertical')
plt.bar(l,hotels_list['Avg_Review'],align='center')

#plt.show()


