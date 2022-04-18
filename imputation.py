# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:05:24 2022

@author: lapflexhome
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import datetime
from functools import reduce
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter


def find_baths(index,listing_df):
    item = listing_df.iloc[index]['description']
    hb_count=0
    fb_count = 0
  #find half-baths
    hb_index = item.find("half-bath")
  #found
    if hb_index != -1:
        hb_check = item[hb_index-2]
        if hb_check.isdigit() == True:
            hb_count = hb_check
        else:
            hb_count = 1
        hb_count = hb_count *0.5
  #find full baths
    fb_index = item.find("full bath")
    if fb_index != -1:
        fb_check = item[fb_index-2]
        if fb_check.isdigit() == True:
            fb_count = fb_check
        else:
            fb_count =1

    total = str(hb_count + fb_count) + " baths"

  #no data found, set to mode (1)
    if total == "0 baths":
        total = "1 bath"
    return total

def find_bedrooms(index,listing_df): #Fill Bedrooms from description
    item = listing_df.iloc[index]['description']
    count=0.0
    index = item.find("bedroom")
   #found
    if index != -1:
        check = item[index-2]
        if check.isdigit() == True:
              count = check
        else:
              count = 1.0
    total = count
  #no data found, set to mode (1)
    if total == 0.0:
        total = 1.0
    return total

def find_beds(index,listing_df):
    item = listing_df.iloc[index]['description']
    count=0.0
    index = item.find("bed")
    #found
    if index != -1:
        check = item[index-2]
        if check.isdigit() == True:
              count = check
        else:
              count = 1.0
    total = count
  #no data found, set to mode (1)
    if total == 0.0:
        total = 1.0
    return total


def data_imputation():
    listing_df=pd.read_csv('listings.csv')
    listing_df=listing_df.drop(columns=["bathrooms","listing_url","scrape_id","last_scraped","picture_url",
                                      "host_url","host_name","host_thumbnail_url","host_picture_url","neighbourhood_group_cleansed",
                                      "calendar_updated","calendar_last_scraped"])

    listing_df['host_response_rate']=listing_df['host_response_rate'].str.replace('%', '')
    x=listing_df['host_response_rate'].dropna().astype(int).mean()
    listing_df['host_response_rate']=listing_df['host_response_rate'].fillna(x).astype(int)

    listing_df['host_acceptance_rate']=listing_df['host_acceptance_rate'].str.replace('%', '')
    x=listing_df['host_acceptance_rate'].dropna().astype(int).mean()
    listing_df['host_acceptance_rate']=listing_df['host_acceptance_rate'].fillna(x).astype(int)

    listing_df['host_response_time'] = listing_df['host_response_time'].fillna(4).astype('category')
    listing_df['host_response_time'].replace(['within an hour', 'within a few hours', 'within a day', 'a few days or more'],
                        [0,1,2,3], inplace=True)

    listing_df['review_scores_rating']=listing_df['review_scores_rating'].fillna(0)
    listing_df['review_scores_cleanliness']=listing_df['review_scores_cleanliness'].fillna(0)
    listing_df['review_scores_checkin']=listing_df['review_scores_checkin'].fillna(0)
    listing_df['review_scores_communication']=listing_df['review_scores_communication'].fillna(0)
    listing_df['review_scores_location']=listing_df['review_scores_location'].fillna(0)

    listing_df.loc[listing_df['bathrooms_text'] == 'Half-bath', 'bathrooms_text'] = '0.5 bath'
    listing_df.loc[listing_df['bathrooms_text'] == 'Shared half-bath', 'bathrooms_text'] = '0.5 shared bath'
    listing_df.loc[listing_df['bathrooms_text'] == 'Private half-bath', 'bathrooms_text'] = '0.5 private bath'

    bathrooms_null_data = listing_df[listing_df['bathrooms_text'].isnull()]
    bath_null_ids=bathrooms_null_data.index.tolist()
    bath_desc_ids = bathrooms_null_data.index[bathrooms_null_data['description'].notnull()].tolist()

    for x in bath_desc_ids:
        baths=find_baths(x,listing_df)
        #print(baths)
        listing_df.at[x, 'bathrooms_text'] = baths

    bath_desc_null_ids = list((Counter(bath_null_ids) - Counter(bath_desc_ids)).elements())
    for x in bath_desc_null_ids:
        listing_df.at[x, 'bathrooms_text'] = "1 bath"

    bedrooms_null_data = listing_df[listing_df['bedrooms'].isnull()]
    bedrooms_null_ids = bedrooms_null_data.index.tolist()
    bedrooms_desc_ids = bedrooms_null_data.index[bedrooms_null_data['description'].notnull()].tolist()

    #loop through indexes with descriptions
    for x in bedrooms_desc_ids:
        bedrooms=find_bedrooms(x,listing_df)
        #print(bedrooms)
        listing_df.at[x, 'bedrooms'] = bedrooms

    #Fill bedrooms data with mode (1)
    bedrooms_desc_null_ids=list((Counter(bedrooms_null_ids)-Counter(bedrooms_desc_ids)).elements())
    for i in bedrooms_desc_null_ids:
        listing_df.at[i,'bedrooms']=1.0

    beds_null_data = listing_df[listing_df['beds'].isnull()]
    beds_null_ids = beds_null_data.index.tolist()
    beds_desc_ids = beds_null_data.index[beds_null_data['description'].notnull()].tolist()

    #loop through indexes with descriptions
    for x in beds_desc_ids:
        beds=find_beds(x,listing_df)
        listing_df.at[x, 'beds'] = beds

    beds_desc_null_ids = list((Counter(beds_null_ids) - Counter(beds_desc_ids)).elements())
    for x in beds_desc_null_ids:
        listing_df.at[x, 'beds'] = 1.0

    # derived column -number of years from host_since
    dt=datetime.datetime(2022, 1, 6, 0, 0)
    listing_df['host_since']=pd.to_datetime(listing_df['host_since'])
    x=dt-listing_df['host_since']
    listing_df['host_since_years']=round(x/datetime.timedelta(days=365),1)
    listing_df['host_since_years']=listing_df['host_since_years'].fillna(0) # 1 missing record

    listing_df['reviews_per_month']=listing_df['reviews_per_month'].fillna(0)
    listing_df['host_is_superhost']=listing_df['host_is_superhost'].fillna('f')
    listing_df['host_total_listings_count']=listing_df['host_total_listings_count'].fillna(0)
    listing_df['host_has_profile_pic']=listing_df['host_has_profile_pic'].fillna('f')
    listing_df['host_identity_verified']=listing_df['host_identity_verified'].fillna('f')
    listing_df['reviews_per_month']=listing_df['reviews_per_month'].fillna(0)
    listing_df['Success_rate']=1-(0.25*listing_df['availability_30']/30+0.25*(listing_df['availability_60']-listing_df['availability_30'])/30+0.25*(listing_df['availability_90']-listing_df['availability_60'])/30+0.25*(listing_df['availability_365']-listing_df['availability_90'])/275)

    calendar_df=pd.read_csv('calendar.csv')
    calendar_df['listing_id']=calendar_df['listing_id'].astype('category')
    calendar_df['date']=pd.to_datetime(calendar_df['date'])
    calendar_df['available']=calendar_df['available'].astype('category')
    calendar_df['price']=calendar_df['price'].str.replace('$','').str.replace(',','').astype(float)
    calendar_df['adjusted_price']=calendar_df['adjusted_price'].str.replace('$','').str.replace(',','').astype(float)
    calendar_df['price_diff']=calendar_df['price']-calendar_df['adjusted_price']


    #create day of week column using "DATE" column
    calendar_df['day_of_week'] = [i.day_name() for i in calendar_df['date'].tolist()]
    calendar_df['day_of_week'] = calendar_df['day_of_week'] .astype('category')

    #extract Number of days booked, 5 Bins based on days booked from calendar data
    df=pd.crosstab(calendar_df.listing_id,calendar_df.available)
    df.columns=['days_booked','days_not_booked']
    df['listing_id']=df.index
    df.reset_index(drop=True,inplace=True)
    df=pd.DataFrame(df,columns=['listing_id','days_booked','days_not_booked'])
    df.drop(columns=['days_not_booked'],inplace=True)
    bins=[0,68,211,305,365]
    df['booking_status']=np.searchsorted(bins, df['days_booked'].values)

    # Extract min/max/mean adjusted price from calendar data
    df1_min_price=calendar_df.groupby('listing_id',as_index=False)['adjusted_price'].min()
    df1_min_price.columns=['listing_id','min_price']
    df1_max_price=calendar_df.groupby('listing_id',as_index=False)['adjusted_price'].max()
    df1_max_price.columns=['listing_id','max_price']
    df1_avg_price=calendar_df.groupby('listing_id',as_index=False)['adjusted_price'].mean()
    df1_avg_price.columns=['listing_id','mean_price']

    #Merge all the calculated columns from calendar data
    tables=[df,df1_min_price,df1_max_price,df1_avg_price]
    dfs = reduce(lambda x,y: pd.merge(x,y,on='listing_id'), tables)

    combined_df=pd.merge(listing_df,dfs,how='inner',left_on='id',right_on='listing_id')
    combined_df.drop('listing_id',axis=1,inplace=True)
    combined_df['mean_price']=combined_df['mean_price'].round(decimals=1)

    model_data=combined_df.copy()

    for i in model_data.index:
        if model_data.loc[i,'host_location']=='San Francisco, California, United States':
            model_data.loc[i,'host_in_sfo']=1
        else:
            model_data.loc[i,'host_in_sfo']=0

    for i in model_data.index:
        if str(model_data.loc[i,'license']).startswith('STR'):
            model_data.loc[i,'licensed']=1
        else:
            model_data.loc[i,'licensed']=0

    return model_data

def hrt_pred():
    model_data=data_imputation()
    #model_data=model_data1.copy()
    model_data.drop(['id','name','host_about','calculated_host_listings_count_entire_homes','review_scores_value','review_scores_accuracy','description','neighborhood_overview','host_id','latitude','longitude','host_since','host_location','host_neighbourhood','host_listings_count','host_verifications','neighbourhood','amenities','price','availability_30','availability_60','availability_90','availability_365','first_review','last_review','license','calculated_host_listings_count','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','days_booked','booking_status','min_price','max_price'],axis=1,inplace=True)
    model_data['host_response_time'].replace(['within an hour', 'within a few hours', 'within a day', 'a few days or more'],
                        [0,1,2,3], inplace=True)

    model_data=model_data.iloc[:,[0,1,2,3,4,5,6,22,31,33,34,35,36,37]]
    model_data=model_data.astype({'licensed':'category',"host_in_sfo":'category',
                               "has_availability":'category',"host_is_superhost":'category',
                                "host_has_profile_pic":'category',"host_identity_verified":'category',
                               "instant_bookable":'category'})
    dummy_columns=['host_is_superhost',
        'host_has_profile_pic',
       'host_identity_verified', 'has_availability',
       'instant_bookable', 'host_in_sfo',
       'licensed']

    for i in dummy_columns:
        if len(model_data.groupby([i]).size())>2:
            encoded_data=pd.get_dummies(model_data,prefix=[i],columns=[i])
    encoded_data=pd.get_dummies(model_data,drop_first=True)

    #split data into train & test(NA's(represented as 4) goes into test)
    test=encoded_data.loc[encoded_data['host_response_time']==4,:]
    train=encoded_data.loc[encoded_data['host_response_time']!=4,:]

    X=train.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]
    Y=train.iloc[:,0]
    X_test=test.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]
    Y_test=test.iloc[:,0]

    X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.1,random_state=1)
    cols=X_train.columns
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_valid=scaler.transform(X_valid)
    X_test=scaler.transform(X_test)
    X_train=pd.DataFrame(X_train,columns=cols)
    X_valid=pd.DataFrame(X_valid,columns=cols)
    X_test=pd.DataFrame(X_test,columns=cols)

    #K=10 gives better accuracy in train.
    knn=KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train.values, Y_train.values)
    Y_pred_test=knn.predict(X_test.values)

    index=test.index.tolist()
    prediction=Y_pred_test
    pred={}
    for i,v in zip(index,prediction):
        pred[i]=v
    df=pd.DataFrame(data=pred,index=[0])
    df=df.T
    #df.columns=['host_response_time']

    return df

def reduce_groups():
    data1=data_imputation()
    x=hrt_pred()
    #impute host response time(NA's are 4) with KNN prediction
    data1.loc[data1['host_response_time']==4,'host_response_time']=x[0].tolist()


    # property type
    pt=data1['property_type'].value_counts()>50
    pt=pt.index[:15] # top 15 indexes have more than 50 counts
    for i in data1.index:
        if data1.loc[i,'property_type'] not in pt:
            data1.loc[i,'property_type']='other'

     #neighbourhood
    neighbourhood1=['Western Addition','Mission','Downtown/Civic Center','Marina','Inner Richmond','Bayview','North Beach',
               'Inner Sunset','Chinatown','Diamond Heights','Russian Hill','Ocean View','Outer Mission','Crocker Amazon',
               'Bernal Heights','Castro/Upper Market','Noe Valley','Potrero Hill','Pacific Heights','Glen Park',
               'Outer Richmond','Parkside','Presidio Heights','Visitacion Valley','Financial District','South of Market',
                'Presidio','Haight Ashbury','Outer Sunset','Twin Peaks','Seacliff']

    neighbourhood2=['Nob Hill','West of Twin Peaks','Excelsior','Lakeshore','Golden Gate Park']
    for i in data1.index:
        if data1.loc[i,'neighbourhood_cleansed'] in neighbourhood1:
            data1.loc[i,'neighbourhood_cleansed']='Group1'
        elif data1.loc[i,'neighbourhood_cleansed'] in neighbourhood2:
            data1.loc[i,'neighbourhood_cleansed']='Group2'

    #bath
    bath_group1=['1 bath', '1.5 baths','2 baths','2.5 baths','3 baths','3.5 baths','4 baths','4.5 baths' ]
    bath_group2=['1 private bath']
    bath_group3=['0.5 shared bath','1.5 shared baths','2 shared baths','2.5 shared baths','3 shared baths','3.5 shared baths',
            '4 shared baths','5 shared baths','5 baths','10 shared baths']
    bath_group4=['0 shared baths','0 baths','0.5 private bath','0.5 bath','1 shared bath','4.5 shared baths','5.5 baths','6 baths','6.5 shared baths'
            ,'10 baths']
    bath_group5=['6 shared baths']
    for i in data1.index:
       if data1.loc[i,'bathrooms_text'] in bath_group1:
           data1.loc[i,'bathrooms_text']='bath group1'
       elif data1.loc[i,'bathrooms_text'] in bath_group2:
           data1.loc[i,'bathrooms_text']='bath group2'
       elif data1.loc[i,'bathrooms_text'] in bath_group3:
           data1.loc[i,'bathrooms_text']='bath group3'
       elif data1.loc[i,'bathrooms_text'] in bath_group4:
           data1.loc[i,'bathrooms_text']='bath group4'
       elif data1.loc[i,'bathrooms_text'] in bath_group5:
           data1.loc[i,'bathrooms_text']='bath group5'

    for i in data1.index:
        if data1.loc[i,'bedrooms']>3:
            data1.loc[i,'bedrooms']='More than 3'

    #amenities
    amenities=[]
    for i in data1['amenities'].values:
        amenity=str(i).replace('[','').replace(']','').replace('"','').replace("'","")
        j=amenity.split(',')
        amenities.extend(j)
    amenities=[i.lower() for i in amenities]

    #get top 40 amenities
    ame_dict={}
    for i in amenities:
        ame_dict[i]=ame_dict.get(i,0)+1
    amenities_sorted={k:v for k,v in sorted(ame_dict.items(),key=lambda item:item[1],reverse=True)}
    top_40_amenities=list(amenities_sorted.keys())[:40]

    #get the count of top 40 amenities in each listing
    top_amenities_count=[]
    for i in data1.index:
        count=0
        x=str(data1.loc[i,'amenities']).lower()
        for i in top_40_amenities:
            if i in x:
                count+=1
        top_amenities_count.append(count)
        #print(top_40_amenities)

    #incorporate a new column for top_amenities_count
    data1['top_amenities_count']=top_amenities_count

    # host verifications
    verifications=[]
    for i in data1['host_verifications'].values:
        verification=str(i).replace('[','').replace(']','').replace('"','').replace("'","").replace(" ","")
        j=verification.split(',')
        verifications.extend(j)
    verifications=[i.lower() for i in verifications]

    #get top 11 amenities
    verif_dict={}
    for i in verifications:
        verif_dict[i]=verif_dict.get(i,0)+1
    verifications_sorted={k:v for k,v in sorted(verif_dict.items(),key=lambda item:item[1],reverse=True)}
    top_verif=list(verifications_sorted.keys())[:11]

    #get the count of top 11 verifications in each listing
    top_verifications_count=[]
    for i in data1.index:
        count=0
        x=str(data1.loc[i,'host_verifications']).lower()
        for i in top_verif:
            if i in x:
                count+=1
        top_verifications_count.append(count)
    data1['host_verification_count']=top_verifications_count

    return data1

def get_n_1_dummies():
    model_data=pd.read_csv('modeldata.csv')
    model_data=model_data.astype({'licensed':'category',"host_in_sfo":'category',
                               "host_is_superhost":'category',
                                "host_has_profile_pic":'category',"host_identity_verified":'category',
                              "instant_bookable":'category','neighbourhood_cleansed':'category',
                             'property_type':'category','room_type':'category',
                             'bathrooms_text':'category','bedrooms':'category'
                             })

    dummy_columns=['host_is_superhost','host_has_profile_pic','host_identity_verified',
               'neighbourhood_cleansed','property_type','room_type','bathrooms_text','bedrooms','instant_bookable',
               'licensed',"host_in_sfo"]

    for i in dummy_columns:
        if len(model_data.groupby([i]).size())>2:
            encoded_data=pd.get_dummies(model_data,prefix=[i],columns=[i],drop_first=True)
    encoded_data=pd.get_dummies(model_data,drop_first=True)
    encoded_data.drop(['Unnamed: 0'],axis=1,inplace=True)
    return encoded_data

def get_n_dummies():
    model_data=pd.read_csv('modeldata.csv')
    model_data=model_data.astype({'licensed':'category',"host_in_sfo":'category',
                               "host_is_superhost":'category',
                                "host_has_profile_pic":'category',"host_identity_verified":'category',
                              "instant_bookable":'category','neighbourhood_cleansed':'category',
                             'property_type':'category','room_type':'category',
                             'bathrooms_text':'category','bedrooms':'category'
                             })

    dummy_columns=['host_is_superhost','host_has_profile_pic','host_identity_verified',
               'neighbourhood_cleansed','property_type','room_type','bathrooms_text','bedrooms','instant_bookable','licensed',"host_in_sfo"]

    for i in dummy_columns:
        if len(model_data.groupby([i]).size())>2:
            encoded_data=pd.get_dummies(model_data,prefix=[i],columns=[i],drop_first=False)
    encoded_data=pd.get_dummies(model_data,drop_first=False)
    encoded_data.drop(['Unnamed: 0'],axis=1,inplace=True)
    return encoded_data
