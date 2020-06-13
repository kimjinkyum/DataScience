# -*- coding: utf-8 -*-

#import library
import warnings
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
import seaborn as sns
from sklearn import linear_model
#read preprocessed data
data=pd.read_excel("../data/preprocess.xlsx",encoding = "ISO-8859-1",sheet_name="first")



#list_column is using column 
list_column=["price","zipcode","property_type","room_type","accommodates","bathrooms","bedrooms","beds","bed_type","number_of_reviews","minimum_nights_check"]
for i in list_column:
    data[i]=preprocessing.scale(np.array(data[i],dtype=float))#normalization by calling sacle



#x, x1 is using to multiple linear regression
x=data[["zipcode","property_type","room_type","accommodates","bathrooms","bedrooms","beds","bed_type","number_of_reviews","minimum_nights_check"]]
x1=data[["zipcode","accommodates","beds","review_scores_rating","property_type"]]

#y is target 
y=data["price"]

#make linearmodel 
regr=linear_model.LinearRegression()
regr1=linear_model.LinearRegression()

#fit data regr: x, reg1: x1
regr.fit(x,y)
regr1.fit(x1,y)
#then predict value
predict=regr.predict(x.to_numpy())
predict1=regr1.predict(x1.to_numpy())


price=(data["price"].to_numpy())
#Calculate SSE, SSR, R
SSE=(np.sum((price-predict)*(price-predict)))
SSE=np.sqrt(SSE)
SSR=np.sum((np.mean(data["price"])-predict)*(np.mean(data["price"])-predict))
SSR=np.sqrt(SSR)
R=SSR/(SSR+SSE)
print("SSE : {0} SSR : {1}".format(SSE,SSR))
print("R: {0}".format(R))



#Calculate SSE, SSR, R
SSE=(np.sum((price-predict1)*(price-predict1)))
SSE=np.sqrt(SSE)
SSR=np.sum((np.mean(data["price"])-predict1)*(np.mean(data["price"])-predict1))
SSR=np.sqrt(SSR)
R=SSR/(SSR+SSE)
print("SSE : {0} SSR : {1}".format(SSE,SSR))
print("R: {0}".format(R))


# draw graph using subplot, scatter

#draw scatter graph of number of reviews to price
plt.subplot(211)
plt.scatter(data["number_of_reviews"],data["price"])
plt.ylim(0,1000)
plt.xlabel("number_of_reviews")
plt.grid()

#draw scatter graph of review scores rating to price
plt.subplot(212)
plt.scatter(data["review_scores_rating"],data["price"])
plt.ylim(0,1000)
plt.xlabel("review_scores_rating")
plt.grid()
plt.show()


#draw scatter graph of minimum nights check to price
plt.scatter(data["minimum_nights_check"],data["price"])
plt.ylim(0,1000)
plt.xlabel("minimum_nights_check")
plt.show()


# draw scatter
#draw scatter graph of property type to price
plt.scatter(data["property_type"],data["price"])
plt.show()
#draw scatter graph of room type to price
plt.scatter(data["room_type"],data["price"])
plt.show()
#draw scatter graph of accommodates to price
plt.scatter(data["accommodates"],data["price"])
plt.show()




