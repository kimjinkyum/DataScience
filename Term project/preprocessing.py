# -*- coding: utf-8 -*-


#import library
import warnings
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from sklearn import linear_model
import math,os,re
import operator

pd.options.mode.chained_assignment = None 

#read orinial data
original=pd.read_csv("../data/data.csv",encoding = "ISO-8859-1")

column=original.columns[:29]


"""
* preprocessingData : preprocess all data by calling function(each column preprocessing)
* input: orinal data
* output:onehot_property_label : sequence of propertytype ,
         onehot_roomtype_label :  sequence of roomtype
         onehot_bedtype_label : sequence of 
         preprocessed_data: data after preprocessing.
"""
def preprocessingData(data):

    #preprocessing city by calling cityPreprocess function.
    data=cityPreprocess(data) 

    #preprocessing host_id, id by calling idPreprocess function.
    data=idPreprocess(data)

    #preprocessing data drop null value in property_type, room_type, bed_type by calling dropna
    data=dropna(data,"property_type")
    data=dropna(data,"room_type")
    data=dropna(data,"bed_type")
    data=data.dropna(subset=["bathrooms","bedrooms","beds","accommodates"])

    #preprocessing zipcode, latitude, longitude by calling zipcodePreprocess, locationPreprocess
    #, zipcodeLocationPreprocess, fillLocation
    data=zipcodePreprocess(data)
    data=locationPreprocess(data)
    data=zipcodeLocationPreprocess(data)
    fillLocation(data)


    #preprocessing bathroom,beds by calling roomPreprocess
    bathroom, beds= roomPreprocess(data)

    #preprocessing property type,room type,bed type by calling oneHotPreprocessing
    # onehot_property/roomtype/bedtype_label: sequence of property/room/bed type
    # onehot_propertyroomtype/bedtype: result of oridinal encoder
    onehot_property_label, onehot_property=OneHotPreprocessing(data,"property_type")
    onehot_roomtype_label,onehot_roomtype=OneHotPreprocessing(data,"room_type")
    onehot_bedtype_label,onehot_bedtype=OneHotPreprocessing(data,"bed_type")

    #preprocessing minimum nights and maximum nights by calling minimumNightPreprocess
    minimumNightNewFeature = minimumNightPreprocess(data)

    #store preprocessed data
    global preprocessed_data


    #copy data to preprocessed_data
    preprocessed_data=data.iloc[:,:]

    #change result to preprocessed
    preprocessed_data["property_type"]=onehot_property
    preprocessed_data["room_type"]=onehot_roomtype
    preprocessed_data["bed_type"]=onehot_bedtype
    preprocessed_data["beds"]=beds
    preprocessed_data["bathrooms"]=bathroom



    colCat=preprocessed_data['amenities'].as_matrix().reshape(-1)
    colCat=str(colCat).split(',')
    trimming(colCat)#특수기호 제거
    colCat=list(dict.fromkeys(colCat))#eliminate duplicated
    amenityCount=findAmenities(colCat)
    amenSize=12
    topTwelve=topNfromDict(amenityCount,amenSize)
    amenX,amenY=[],[]#생략하기..
    for i in range(amenSize):
        amenX.append(topTwelve[i][0])
        amenY.append(topTwelve[i][1])
    #[['Wifi','Internet'],['Heating'],['essentials','shampoo','dryer'],['kitchen'],['elevator']]
    amenDic={'Wifi':0,'Heating':0,'Showering':0,'Kitchen':0,'Elevator':0}
    amenDic['Wifi']=amenY[0]+amenY[9]#Wifi+Internet
    amenDic['Heating']=amenY[1]
    amenDic['Showering']=amenY[2]+amenY[7]+amenY[8]#Essentials+Shampoo+Dryer
    amenDic['Kitchen']=amenY[3]
    amenDic['Elevator']=amenY[10]
    OHamenity=oneHotAmenities()
    OH=['Wifi','Heating','Shower','Kitchen','Elevator']
    revOHamenity=np.array(OHamenity).T
    for i in range(len(revOHamenity)):
        preprocessed_data[OH[i]]=pd.Series(revOHamenity[i])
    preprocessed_data=pd.concat([preprocessed_data, minimumNightNewFeature], axis = 1)
    preprocessed_data=preprocessed_data.drop(["guests_included","number_of_reviews_ltm","last_review","reviews_per_month","minimum_nights","maximum_nights","latitude","longitude","host_response_time","host_response_rate","amenities"],axis=1)
    preprocessed_data=weekPricePredict(preprocessed_data)
    preprocessed_data=monthPricePredict(preprocessed_data)
    preprocessed_data = reviewRatingPreProcess(preprocessed_data)
    print(preprocessed_data.head())

    #return sequence of oridinal label, then result of preprocessing
    return onehot_property_label,onehot_roomtype_label,onehot_bedtype_label,preprocessed_data
    


"""
* cityPreprocess : preprocess city column ( drop if it is not "Asterdam" and missing)
* input: orinal data
* output: preprocessed_data: data after preprocessing.
"""
def cityPreprocess(data):
    #drop if city is missing value
    data_dropna=dropna(data,"city")
    #drop if city is not "Asterdam"
    data_city=data_dropna[data_dropna["city"]=="Amsterdam"]
    #reassign inde
    data_city=data_city.reset_index(drop=True)
    #return preprocessed data
    return data_city



"""
* idPreprocess : preprocess id, host id column ( drop if it missing and wrong type)
* input: orinal data
* output: preprocessed_data: data after preprocessing.
"""
def idPreprocess(data):
    # drop value if id or host_id is missing value
    data=dropna(data,"id")
    data=dropna(data,"host_id")
    # drop value if it has wrong type by calling dropDigit
    data=dropDigit(data,"id",False)
    data=dropDigit(data,"host_id",False)
    #return preprocessed data
    return data

"""
* zipcodePreprocess : preprocess zipcode column(change format, if it is out of range then change missing value)
* input: orinal data
* output: preprocessed_data: data after preprocessing.
"""
def zipcodePreprocess(origin):
    #find index zipcode
    index_zipcode=findIndex("zipcode",origin)


    #change format to have only number so if value include string then remove string

    for i in range(len(origin)):
        list_temp=[]# store part of string
        zipcode=str(origin.iloc[i,index_zipcode])
        if zipcode=="nan":
            continue;
            
        
        else:
            for j in range(len(zipcode)):
                if ord(zipcode[j])>=48 and ord(zipcode[j])<=57:
                    list_temp.append(j)# if value include string then sotre list_temp
                    
                if j==len(zipcode)-1 and len(list_temp)>0:
                    zipcode=zipcode[:len(list_temp)]#then only use part of number.
                
                    
        origin.iat[i,index_zipcode]=zipcode #Chnage original value 

    #if it doesn' match format, out of range then change missing value.   
    for i in range(len(origin)):
        zipcode=str(origin.iloc[i,index_zipcode])
        if zipcode=="nan":
            continue;
            
        else:
            if is_digit(zipcode)==False:#check right format( only number)
                zipcode="nan"
            else:
                if int(zipcode)<1011 or int(zipcode)>1109:#check range
                    zipcode="nan"                
        origin.iat[i,index_zipcode]=zipcode#Chnage original value       
    origin=origin.reset_index(drop=True)#reassign index
    return origin#return preprocessed data



"""
* locationPreprocess : preprocess longitude,latitude column(if it is out of range or wrong type then change missing value)
* input: orinal data
* output: preprocessed_data: data after preprocessing.
"""
        
def locationPreprocess(data):
    #find index of latitude , longitude
    index_latitude=findIndex("latitude",data)
    index_longitude=findIndex("longitude",data)

    #latitude
    for i in range(len(data)):
        latitude=(data.iloc[i,index_latitude])

        #if it isn't right type(not number)
        if is_digit(latitude)==False:
            latitude=np.NAN
            latitude=float(latitude)
            # then change original number to np.nan
            data.iat[i,index_latitude]=latitude
            
        else:
            latitude=float(latitude)
            #if it is out of range 
            if latitude<52.28 or latitude>54:
                latitude=np.NAN
                latitude=float(latitude)
                # then change original number to np.nan
                data.iat[i,index_latitude]=latitude

    #longitude
    for i in range(len(data)):
        longitude=(data.iloc[i,index_longitude])
        #if it isn't right type(not number)
        if is_digit(longitude)==False:
            longitude=np.NAN
            longitude=float(longitude)
            # then change original number to np.nan
            data.iat[i,index_longitude]=longitude
       
        else:
            longitude=float(longitude)
            #if it is out of range 
            if longitude<4.7 or longitude>5.0:
                # then change original number to np.nan
                longitude=np.NAN
                data.iat[i,index_longitude]=longitude
        
         
    #reassign index
    data=data.reset_index(drop=True)
    #return preprocessed data
    return data

"""
* zipcodeLocationPreprocess : preprocess zipcode, latitude, longitude column(if all of them is missing value then drop it)
* input: orinal data
* output: preprocessed_data: data after preprocessing.
"""
def zipcodeLocationPreprocess(data):
    #find index zipcode,latitude, longitude
    index_zipcode=findIndex("zipcode",data)
    index_latitude=findIndex("latitude",data)
    index_longitude=findIndex("longitude",data)
    list_drop=[]#store index of drop
    
    for i in range((len(data))):
        zipcode=data.iloc[i,index_zipcode]
        zipcode=str(data.iloc[i,index_zipcode])
        latitude=data.iloc[i,index_latitude]
        longitude=data.iloc[i,index_longitude]
        latitude=float(latitude)
        longitude=float(longitude)
        #if all is missing value
        if zipcode=="nan":
            if np.isnan(latitude):
                if np.isnan(longitude):
                    #add index to list_drop
                    list_drop.append(i)
    #store drop data to data
    data=data.drop(list_drop)
    #reassign index
    data=data.reset_index(drop=True)
    #return preprocessed data
    return data
"""
* fillLocation : preprocess zipcode, latitude, longitude column(fill missing value)
* input: orinal data
* output: preprocessed_data: data after preprocessing.
"""
def fillLocation(data):

    #find index zipcode, latitude, longitude
    index_zipcode=findIndex("zipcode",data)
    index_latitude=findIndex("latitude",data)
    index_longitude=findIndex("longitude",data)

    #fill missing value by calling findNearst, fill function
    for i in range(len(data)):
        #read data
        zipcode=data.iloc[i,index_zipcode]
        zipcode=str(data.iloc[i,index_zipcode])
        latitude=data.iloc[i,index_latitude]
        longitude=data.iloc[i,index_longitude]
        latitude=float(latitude)
        longitude=float(longitude)
        #list temp is store already used index
        list_temp=[i]
        #if zipcode is nan then fill zipcdoe
        if zipcode=="nan":
            # if latitude is nan, then using longitude to fill data
            if np.isnan(latitude):
                #find near_index by calling findNearest
                near_index=findNearst(data,i,"longitude",list_temp)
                #fill missing value by calling fill
                data=fill(data,i,"longitude","zipcode",near_index)
            # if longitude is nan, then using latitude to fill data
            else:
                #find near_index by calling findNearest
                near_index=findNearst(data,i,"latitude",list_temp)
                #fill missing value by calling fill
                data=fill(data,i,"latitude","zipcode",near_index)
                
    for i in range(len(data)):
        latitude=data.iloc[i,index_latitude]
        longitude=data.iloc[i,index_longitude]
        latitude=float(latitude)
        longitude=float(longitude)
        #list temp is store already used index
        list_temp=[i]
        
        # if longitude is nan, then using zipcode fill data
        if np.isnan(latitude):
                #find near_index by calling findNearest
                near_index=findNearst(data,i,"zipcode",list_temp)
                #fill missing value by calling fill
                data=fill(data,i,"zipcode","latitude",near_index)
        # if longitude is nan, then using zipcode fill data   
        if np.isnan(longitude):
                #find near_index by calling findNearest
                near_index=findNearst(data,i,"zipcode",list_temp)
                #fill missing value by calling fill
                data=fill(data,i,"zipcode","latitude",near_index)


"""
* findNearst : findNearst elements
* input: data: original data, index: using index, target: target index, list_temp: already used index
* output: index_near: index of nearest elements
"""
def findNearst(data,index,target,list_temp):
    #find index of target
    target_index=findIndex(target,data)
    #read data
    nearset_target=float(data.iloc[index,target_index])
    diff=1000
    index_near=0
    #find nearest data(difference is smallest) except null, already used index
    for i in range(len(data)):
        
        if np.isnan(float(data.iloc[i,target_index])):
            continue;
        elif i in list_temp:
            continue;
            
        else:
            #print(np.abs(float(data.iloc[i,target_index])-nearset_target))
            if np.abs(float(data.iloc[i,target_index])-nearset_target)<diff:
                index_near=i
                diff=np.abs(float(data.iloc[i,target_index])-nearset_target)

    #return index of nearest
    return index_near

"""
* fill : fill missing value
* input: data: original data, index: using index to fill, target: target,
*        feature: column which is filled, index_nearest: already used index 
* output: preprocessed_data: data after preprocessing.
"""
def fill(data, index, target,feature,index_near):
    feature_index=findIndex(feature,data)
    
    if np.isnan(float(data.iloc[index_near,feature_index])):
        list_temp=[index_near] #store to already used index
        #  if nearest index value is not missing value
        while(True):
            #find again nearest index
            index_near=findNearst(data,index,target,list_temp)
            #if it is null then store list_temp 
            if np.isnan(float(data.iloc[index_near,feature_index])):
                list_temp.append(index_near)
            else:
                break;
    #change original data        
    data.iat[index,feature_index]=data.iloc[index_near,feature_index]
    #print(index," ",data.iloc[index,feature_index])
    #return preprocessed data
    return data


"""
* dropna : drop row if it is null value
* input: origin: original data, feature:used feature
* output: preprocessed_data: data after preprocessing.
"""
def dropna(origin,feacture):
    #drop null value
    origin=origin.dropna(how="all",subset=[feacture])
    #reassign index
    origin=origin.reset_index(drop=True)
    #return preprocessed data
    return origin
"""
* dropna : drop row if it is digit or not digit
* input: origin: original data, target: used feature
*        what: true: drop digit, false; drop if it is not digit
* output: preprocessed_data: data after preprocessing.
"""
def dropDigit(data,target,what):
    #find index of target
    index=findIndex(target,data)
    #store index to delete
    list_digit=[]
    #what is true then drop if it digit
    if what==True:
        for i in range(len(data)):
            #check it is digit by calling is_digit
            if is_digit(data.iloc[i,index])==True:
                list_digit.append(i)#add index to list_digit
    #what is else then drop if it not digit
    else:
        for i in range(len(data)):
            #check it is digit by calling is_digit
            if is_digit(data.iloc[i,index])==False:
                list_digit.append(i)#add index to list_digit
    #drop list_digit            
    data=data.drop(list_digit)
    #reassign data
    data=data.reset_index(drop=True)
    #return preprocessed data
    return data

"""
* findindex : find index of target column
* input: target: feature wanted to find index ,data: original data
* output: index: -1 meaning fail
"""
def findIndex(target,data):
    column_list=data.columns# column list store all of columns

    for i in range(len(column_list)):
        #if target exist in column_list then return this index
        if column_list[i]==target:
            return i
    # else then return -1
    return -1

"""
* is_digit : check it is digit
* input: temp: value wanted to check
* output: True- it is digit , False- it is not digit
"""
def is_digit(temp):
    temp=str(temp)
    #try to convert float to temp
    try:
        #if success then return True
        float(temp)
        return True
    #if fail, then return False
    except ValueError:
        return False

"""
* OneHotPreprocess : preprocess categorical value
* input: orinal data,target: feature wanted to onehot Encoder
* output: onehot.columns: sequence of oridinal list: result 
"""
def OneHotPreprocessing(data,target):

    #drop null and digit
    drop_na=dropna(data,target)
    drop_na=dropDigit(drop_na, target,True)

    #one hot encode to calling by get_dumnies
    if target=="property_type":
        one_hot=pd.get_dummies(drop_na.property_type)
    elif target=="room_type":
        one_hot=pd.get_dummies(drop_na.room_type)
    elif target=="bed_type":
        one_hot=pd.get_dummies(drop_na.bed_type)     

    #sequence of oridinal list by calling index
    lis=index(one_hot,one_hot.columns)
    #return one_hot.columns, list
    return one_hot.columns,lis 

"""
* index: sequence of column
* input: orinal data, column
* output: index: sequence of column.
"""
def index(data,column):
    index=[]#store index
    for i in range(len(data)):
        for j in range(len(column)):
            #i data is 1
            if data.iloc[i,j]==1:
                index.append(j)#add index to index
    return index

"""
* converFloat : convert float to integer
* input: orinal data, target; wanted to conver
* output: data: data after converting.
"""
def convertFloat(data, target):
    #find index target
    index=findIndex(target,data)
    for i in range(len(data)):
        #if value is not integer type
        if float(data.iloc[i,index]).is_integer()==False:
            #convert to integer using round
            data.iat[i,index]=round(float(data.iloc[i,index]))
    #return data after converting
    return data

"""
* roomPreprocess : preprocess bathrooms, bedrooms, beds, acoomodates column(change format, drop if it is missing value, add column)
* input: orinal data
* output: bathrooms_accom:new column bathroom/accommodates , beds_accom: new column beds/accommodates
"""
def roomPreprocess(data):
    index_bath=findIndex("bathrooms",data)
    index_bed=findIndex("beds",data)
    index_acom=findIndex("accommodates",data)
    list_type=["bathrooms","bedrooms","beds","accommodates"]

    #drop missing value , wrong value(type error)
    for i in range(len(list_type)):
        data=dropDigit(data,list_type[i],False)
        data=dropna(data,list_type[i])
    
    # convert wrong value(float) by calling covertFloat
    for i in range(len(list_type)):
        data=convertFloat(data,list_type[i])
    bathrooms_accom=[]#store new column
    #calculate bathroom divide accommodates
    for i in range(len(data)):
        bathrooms_accom.append(float(int(data.iloc[i,index_bath])/int(data.iloc[i,index_acom])))
    beds_accom=[]
    
    #calculate beds divide accommodates
    for i in range(len(data)):
        beds_accom.append(float(int(data.iloc[i,index_bed])/int(data.iloc[i,index_acom])))
    return bathrooms_accom, beds_accom
def minimumNightPreprocess(data):
    night = data["minimum_nights"]

    temp=(data.iloc[:,22].to_numpy())
    temp=temp.astype(np.float32)

    for i in range(len(night)): # outlier detection
        if((float(night[i]) > np.mean(temp) + 3 * np.std(temp)) or (float(night[i]) < np.mean(temp) -  3 * np.std(temp))):
            night[i] = np.mean(temp)
    # print(night)
    night = pd.DataFrame({'minimum_nights': night, 'minimum_nights_check': 0})
    night["minimum_nights_check"] = 0

    for i in range(len(night)): # minimum night가 작으면 0 크면 1로 feature creation
        if(float(night.iloc[i,0]) >= 5):
            night.iloc[i,1] = 1
        else:
            night.iloc[i,1] = 0

    return night["minimum_nights_check"]


def weekPricePredict(datat):
    price_week = datat[["price", "weekly_price"]]
    price_origin = datat["price"]
    week_origin = datat["weekly_price"]
    
    price_week_noNan = price_week.dropna() # drop nan from week
    price1 = price_week_noNan["price"]
    week = price_week_noNan["weekly_price"]

    reg = linear_model.LinearRegression() # price and week linear regression
    reg.fit(price1[:, np.newaxis], week)

    pw = reg.predict(price_origin[:,np.newaxis]) #predicted week_price
    
    for i in range(len(price_origin)):  # enter predicted data at week nan data
        w = float(week_origin[i])
        if(np.isnan(w)):
            datat["weekly_price"][i] = int(pw[i])

    return datat

# month 가격 채워넣기
def monthPricePredict(datat):
    price_month = datat[["price", "monthly_price"]]
    price_origin = datat["price"]
    month_origin = datat["monthly_price"]

    price_month_noNan = price_month.dropna() # drop nan from month
    price2 = price_month_noNan["price"]
    month = price_month_noNan["monthly_price"]

    reg2 = linear_model.LinearRegression() # price and month linear regression
    reg2.fit(price2[:, np.newaxis], month)

    pm = reg2.predict(price_origin[:,np.newaxis]) #predicted month_price

    for i in range(len(price_origin)):  # enter month nan data to predicted data
        m = float(month_origin[i])
        if(np.isnan(m)):
            datat["monthly_price"][i] = int(pm[i])

    return datat
#####'amenities'분리해주는 functions######
#####'amenities'분리해주는 functions######
def trimming(strList):
    for i in range(len(strList)):
        #trim(strList[i])
        strList[i]=splitString(strList[i])
#str양끝에 특수기호가 있으면 벗겨줌
def trim(str):
    while(not(str.isalnum()) and not(splitString(str).isalnum())):
        str=str.strip()
        str=str.strip(']')
        str=str.strip('[')
        str=str.strip(':')
        str=str.strip('{')
        str=str.strip('}')
        str=str.strip('\'')
        str=str.strip('"')
        str=str.strip('_')
        str=str.strip('-')
        str=str.strip('\"')
        str=str.strip('}')
#str을에서 특수기호 기준으로 분리해줌 
def splitString(str):
    tmp=str
    tmp="".join(tmp.split())
    tmp="".join(tmp.split('/'))
    tmp="".join(tmp.split('-'))
    tmp="".join(tmp.split(':'))
    tmp="".join(tmp.split('.'))
    tmp="".join(tmp.split('_'))
    tmp="".join(tmp.split('#'))
    tmp="".join(tmp.split('"'))
    tmp="".join(tmp.split('\''))
    tmp="".join(tmp.split('{'))
    tmp="".join(tmp.split('}'))
    tmp="".join(tmp.split('['))
    tmp="".join(tmp.split(']'))
    tmp="".join(tmp.split('('))
    tmp="".join(tmp.split(')'))
    tmp="".join(tmp.split('<'))
    tmp="".join(tmp.split('>'))
    return tmp
#유니크한 amenity들의 개수 세기 
def findAmenities(colCat):
    amenitycount={colCat[0]:0}
    for i in range(len(colCat)):
        amenitycount[colCat[i]]=0
    for i in range(len(preprocessed_data)):
        for x in amenitycount:
            if str(preprocessed_data['amenities'][i]).find(x)>0:
                amenitycount[x]=amenitycount[x]+1
    return amenitycount;#return: {'amenity이름':amenity의 전체빈도수}

#dic(dictionary) 중에서 빈도수가 가장높은 n개만 뽑아서 큰순서로 리턴
def topNfromDict(dic,n):
    topN=[]
    sortedDict=sorted(dic.items(),key=operator.itemgetter(1),reverse=True)#sort by value in reverse
    for i in range(n):
        item=sortedDict[i]
        topN.append(item)
    return topN #return[('key',value)*n]array
def oneHotAmenities():
    OneHotAmenities=[]
    for i in range(len(preprocessed_data)):
        count=[0,0,0,0,0]
        if re.search('Wifi'or'Internet',preprocessed_data['amenities'].iloc[i],re.IGNORECASE):
            count[0]=count[0]+1
        if re.search('Heating',preprocessed_data['amenities'].iloc[i],re.IGNORECASE):
            count[1]=count[1]+1
        if re.search('essentials'or'shampoo'or'dryer',preprocessed_data['amenities'].iloc[i],re.IGNORECASE):
            count[2]=count[2]+1
        if re.search('kitchen',preprocessed_data['amenities'].iloc[i],re.IGNORECASE):
            count[3]=count[3]+1
        if re.search('elevator',preprocessed_data['amenities'].iloc[i],re.IGNORECASE):
            count[4]=count[4]+1
        OneHotAmenities.append(count)
    return OneHotAmenities
def reviewRatingPreProcess(data):
    review_rating = data["review_scores_rating"]

    for i in range(len(review_rating)):
        r = float(review_rating[i])
        if(np.isnan(r)): # 빈 값이면 0으로 채우기
            data["review_scores_rating"][i] = 0

    return data
lable1,labe2,label3,preprocessed_data=preprocessingData(original)

print(preprocessed_data.columns)

preprocessed_data.to_excel("../data/preprocess.xlsx",sheet_name="first")










