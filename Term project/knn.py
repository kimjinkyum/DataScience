#KNN###
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
import numpy as np

preprocessed_data=pd.read_excel("../data/preprocess.xlsx",encoding = "ISO-8859-1")
index=list(range(len(preprocessed_data)))
print(len(preprocessed_data)*0.7)
random.shuffle(index)


trainingSize,testSize=int(len(preprocessed_data)*0.7),int(len(preprocessed_data)-(len(preprocessed_data)*0.7))

print(trainingSize,testSize)
preprocessed_X=preprocessed_data.drop(['id','host_id','neighbourhood_cleansed','city','price','weekly_price','cleaning_fee'],axis=1)
#preprocessed_X=preprocessed_data.drop(['zipcode','accommodates','beds','review_scores_rating','id','host_id','neighbourhood_cleansed','city','price','weekly_price','cleaning_fee'],axis=1)
#preprocessed_X=preprocessed_data[['zipcode','accommodates','beds','review_scores_rating']]

trainingX=preprocessed_X.iloc[index[0:trainingSize]]
preprocessed_Y=preprocessed_data["price"]
"""
testX=[]
for i in range(10):
    testX.append(preprocessed_X.iloc[index[trainingSize+i*testSize:trainingSize+(i+1)*testSize-1]])

"""
testX=preprocessed_X.iloc[index[trainingSize:]]

print(testX.columns)
print(trainingX.columns)
trainingY=preprocessed_Y.iloc[index[0:trainingSize]]
"""
testY=[]
for i in range(10):
    testY.append(preprocessed_Y.iloc[index[trainingSize+i*testSize:trainingSize+(i+1)*testSize-1]])
"""
clf=KNeighborsRegressor(n_neighbors=15)
clf.fit(trainingX,trainingY)
testY=clf.predict(testX)
print(testY)
accuracy=[]
print("Considered data columns: ")
print(preprocessed_X.columns)
#print("K-neighbor accuracy: ",clf.score(trainingX,trainingY))

print("K-neighbor accuracy: \n")
score=cross_val_score(clf,trainingX,trainingY,cv=10)
print(score,"\n",np.mean(score))
