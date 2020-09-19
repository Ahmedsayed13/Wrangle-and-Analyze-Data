#!/usr/bin/env python
# coding: utf-8

# ### Data Wrangling Project by Ahmed Sayed

# In[1]:


import pandas as pd
import numpy as np
import tweepy
import requests
import json
import re
import matplotlib.pyplot as plt
import datetime
import os
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Gather Data
# - Open 'twitter-archive-enhanced.csv' and store it in a dataframe

# In[2]:


# Store archive csv file in a dataframe
archive = pd.read_csv('twitter-archive-enhanced.csv')


# - Download image-predictions tsv file from the provided url and store in a dataframe

# In[3]:


# Download image-predictions tsv file
url='https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
file_name=url.split('/')[-1]
response = requests.get(url)
if not os.path.isfile(file_name):
    with open(file_name,'wb') as f:
        f.write(response.content)


# In[4]:


# Store image-predictions tsv file in a dataframe
image_prediction=pd.read_csv('image-predictions.tsv',sep='\t')


# In[26]:


# Twitter API data
import tweepy
from tweepy import OAuthHandler
import json
from timeit import default_timer as timer

# Query Twitter API for each tweet in the Twitter archive and save JSON in a text file
# These are hidden to comply with Twitter's API terms and conditions
consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

# NOTE TO STUDENT WITH MOBILE VERIFICATION ISSUES:
# df_1 is a DataFrame with the twitter_archive_enhanced.csv file. You may have to
# change line 17 to match the name of your DataFrame with twitter_archive_enhanced.csv
# NOTE TO REVIEWER: this student had mobile verification issues so the following
# Twitter API code was sent to this student from a Udacity instructor
# Tweet IDs for which to gather additional data via Twitter's API
tweet_ids = df_1.tweet_id.values
len(tweet_ids)

# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
count = 0
fails_dict = {}
start = timer()
# Save each tweet's returned JSON as a new line in a .txt file
with open('tweet_json.txt', 'w') as outfile:
    # This loop will likely take 20-30 minutes to run because of Twitter's rate limit
    for tweet_id in tweet_ids:
        count += 1
        print(str(count) + ": " + str(tweet_id))
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            print("Success")
            json.dump(tweet._json, outfile)
            outfile.write('\n')
        except tweepy.TweepError as e:
            print("Fail")
            fails_dict[tweet_id] = e
            pass
end = timer()
print(end - start)
print(fails_dict)


# 
# 

# - Convert tweet_json.txt file to a datarame that contains (tweet_id, retweet_count, favorite_count) columns

# In[5]:


df_list=list()
with open('tweet_json.txt','r') as file:
    for line in file:
        tweet=json.loads(line)
        tweet_id=tweet['id']
        retweet_count=tweet['retweet_count']
        favorite_count=tweet['favorite_count']
        df_list.append({'tweet_id':tweet_id,'retweet_count':retweet_count,'favorite_count':favorite_count})
        
api_df=pd.DataFrame(df_list)
api_df.head(5)


# ### Assess Data
# - First Visual Assessment
# 

# In[6]:


archive.head(30)


# In[7]:


archive.sample(5)


# In[8]:


image_prediction.head(30)


# In[9]:


api_df.head(20)


# - Second Programmatic Assessment
# - archive dataframe 

# In[10]:


archive.rating_numerator.describe()


# In[11]:


archive.rating_numerator.value_counts()


# In[12]:


archive.rating_denominator.value_counts()


# In[13]:


archive.info()


# In[14]:


archive.doggo.value_counts()


# In[15]:


archive.floofer.value_counts()


# In[16]:


archive.pupper.value_counts()


# In[17]:


archive.puppo.value_counts()


# ### Quality issues in archive 
# - tweet_id data type must be object.
# - timestamp data type must be datetime.
# - not required data in 'retweeted_status_id','retweeted_status_user_id' and      'retweeted_status_timestamp' columns.
# - not required data 'in_reply_to_status_id' ,'in_reply_to_user_id columns'.
# - some wrong values at rating_numerator and rating_denominator
# - missing values in expanded urls
# -------------------------

# - Image_prediction dataframe

# In[18]:


image_prediction.shape


# In[19]:


image_prediction.info()


# ### Quality issues in image_prediction
# - tweet_id data must be object
# - titles of p1 ,p1_conf,p1_dog,p2 ,p2_conf,p2_dog, p3 ,p3_conf,p3_dog have no meaning.
# 
# ### Tidiness issues in image_prediction
# - there is more than one column that represents the prediction values
# 

# - api_df

# In[20]:


api_df.shape


# In[21]:


api_df.info()


# ### Tidiness issues in api_df 
# - this table must be merged with the archive dataframe and then merge the two data frames with archive data frame to make the master dataframe

# ## Data Cleaning
# ### archive dataframe quality issues
# 

# In[22]:


### Before cleaning any data take a copy from the original dataframes
archive_clean=archive.copy()
imprediction_clean=image_prediction.copy()
api_new=api_df.copy()


# ### Define
# - change data type of tweet_id to object using astype
# - change data type of timestamp to datetime
# 

# ### Code

# In[23]:



archive_clean['tweet_id']=archive_clean['tweet_id'].astype(object)
archive_clean['timestamp']=pd.to_datetime(archive_clean['timestamp'])


# ### Test

# In[24]:


archive_clean.info()


# ### Define
# - Drop in_reply_to_status_id, in_reply_to_user_id, retweeted_status_id, retweeted_status_user_id, retweeted_status_timestamp

# ### Code

# In[25]:


archive_clean.drop(['in_reply_to_status_id','in_reply_to_user_id'],axis=1,inplace=True)


# In[26]:


archive_clean.drop(['retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp'],axis=1,inplace=True)


# ### Test

# In[27]:


archive_clean.info()


# ### Define
# - change all the values of archive_new['rating_denominator'] to 10
# - change the worng values of archive_new['rating_numerator'] to more reasonable numbers

# ### Code

# In[28]:


archive_clean['rating_denominator']=10


# In[29]:


archive_clean.rating_numerator.replace(9,10,inplace=True)
archive_clean.rating_numerator.replace(8,18,inplace=True)
archive_clean.rating_numerator.replace(7,17,inplace=True)
archive_clean.rating_numerator.replace(6,16,inplace=True)
archive_clean.rating_numerator.replace(5,15,inplace=True)
archive_clean.rating_numerator.replace(4,14,inplace=True)
archive_clean.rating_numerator.replace(3,13,inplace=True)
archive_clean.rating_numerator.replace(1,10,inplace=True)
archive_clean.rating_numerator.replace(2,12,inplace=True)
archive_clean.rating_numerator.replace(420,12,inplace=True)
archive_clean.rating_numerator.replace(0,10,inplace=True)
archive_clean.rating_numerator.replace(75,17,inplace=True)
archive_clean.rating_numerator.replace(80,18,inplace=True)
archive_clean.rating_numerator.replace(20,19,inplace=True)
archive_clean.rating_numerator.replace(24,19,inplace=True)
archive_clean.rating_numerator.replace(26,19,inplace=True)
archive_clean.rating_numerator.replace(44,19,inplace=True)
archive_clean.rating_numerator.replace(50,19,inplace=True)
archive_clean.rating_numerator.replace(60,16,inplace=True)
archive_clean.rating_numerator.replace(165,16,inplace=True)
archive_clean.rating_numerator.replace(84,18,inplace=True)
archive_clean.rating_numerator.replace(88,18,inplace=True)
archive_clean.rating_numerator.replace(143,14,inplace=True)
archive_clean.rating_numerator.replace(182,18,inplace=True)
archive_clean.rating_numerator.replace(204,19,inplace=True)
archive_clean.rating_numerator.replace(666,16,inplace=True)
archive_clean.rating_numerator.replace(960,19,inplace=True)
archive_clean.rating_numerator.replace(1776,17,inplace=True)
archive_clean.rating_numerator.replace(27,17,inplace=True)
archive_clean.rating_numerator.replace(45,14,inplace=True)
archive_clean.rating_numerator.replace(99,19,inplace=True)
archive_clean.rating_numerator.replace(121,12,inplace=True)
archive_clean.rating_numerator.replace(144,14,inplace=True)


# ### Test

# In[30]:


archive_clean.rating_numerator.value_counts()


# In[31]:


archive_clean.rating_denominator.value_counts()


# ### Define
# - remove rows with missing values in expanded urls

# In[32]:


archive_clean.isna().sum()


# ### Code

# In[33]:


archive_clean.dropna(inplace=True)


# ### Test

# In[34]:


archive_clean.isna().sum()


# In[35]:


archive_clean.info()


# ### imprediction_clean quality issues

# ### Define
# - change time_id column data type to object
# - change title of p1 to prediction_1, p1_conf to prediction_1confidence , p1_dog to prediction1_dog 
# - apply of these changes in both p2 and p3
# 

# ### Code

# In[36]:


imprediction_clean['tweet_id']=imprediction_clean['tweet_id'].astype(object)
imprediction_clean.rename(columns={'p1':'prediction1','p1_conf':'prediction1_confidence','p1_dog':'prediction1_dog'},inplace =True)
imprediction_clean.rename(columns={'p2':'prediction2','p2_conf':'prediction2_confidence','p2_dog':'prediction2_dog'},inplace =True)
imprediction_clean.rename(columns={'p3':'prediction3','p3_conf':'prediction3_confidence','p3_dog':'prediction3_dog'},inplace =True)


# ### Test

# In[37]:


imprediction_clean.info()


# ### imprediction_clean tidiness issues
# ### Define
# - predictions value should be represented by only 1 column so i will take the one with the highest probability

# ### Code

# In[38]:


imprediction_clean.drop(['prediction2','prediction2_confidence','prediction2_dog'],axis=1,inplace=True)
imprediction_clean.drop(['prediction3','prediction3_confidence','prediction3_dog'],axis=1,inplace=True)


# ### Test

# In[39]:


imprediction_clean.info()


# ### api_new tidiness issue
# ### Define
# - api_new should be merged with imprediction_clean table

# ### Code

# In[40]:


api_new.info()


# In[41]:


# see the difference between number of tweet id in both data frames
not_shared = (~api_new.tweet_id.isin(list(imprediction_clean.tweet_id)))
not_shared.sum()


# In[42]:


# remove the unshared tweet id from api_new
api_new = api_new[~not_shared]


# In[43]:


api_new.info()


# In[44]:


api_new['tweet_id']=api_new['tweet_id'].astype(object)


# In[45]:


master_df_1=pd.merge(api_new,imprediction_clean,on="tweet_id")


# In[46]:


master_df_1.info()


# In[47]:


twitter_archive_master=pd.merge(master_df_1,archive_clean,on='tweet_id')


# ### Some cleaning for the master dataframe

# In[48]:


twitter_archive_master.info()


# In[49]:


twitter_archive_master.head()


# In[50]:


twitter_archive_master.prediction1.nunique()


# In[51]:


twitter_archive_master=twitter_archive_master[twitter_archive_master['prediction1_dog']== True]


# In[52]:


twitter_archive_master.prediction1.nunique()


# In[57]:


twitter_archive_master.head()


# In[53]:


twitter_archive_master.name.value_counts()


# In[54]:


twitter_archive_master.info()


# - There is no added value for our analysis from prediction1_confidence and prediction1_dog, these data we took from the provided file that was made by a machine learning algorithm to tell us wether this image has a dog or not and tells us the dog breed if it is a dog, it provided three predictions and i took the prediction with the highest probability to be right

# In[55]:


twitter_archive_master.drop(['prediction1_confidence','prediction1_dog'],axis=1,inplace=True)


# In[56]:


twitter_archive_master.info()


# In[57]:


twitter_archive_master.head()


# - It would also make more sense if we change the column header of prediction1 to dog breed

# In[58]:


twitter_archive_master.rename(columns={'prediction1':'dog_breed'},inplace=True)


# In[59]:


twitter_archive_master.head()


# In[60]:


twitter_archive_master.info()


# In[61]:


twitter_archive_master.dog_breed.nunique()


# In[62]:


twitter_archive_master.dog_breed.unique()


# In[63]:


twitter_archive_master.dog_breed.duplicated()


# In[64]:


twitter_archive_master.dog_breed.drop_duplicates(inplace=True)


# In[65]:


twitter_archive_master.dog_breed.nunique()


# In[66]:


twitter_archive_master.to_csv('twitter_archive_master.csv',index=False)


# In[67]:


twitter_archive_master.dog_breed.nunique()


# In[68]:


twitter_archive_master.retweet_count.describe()


# In[72]:


twitter_archive_master[twitter_archive_master.retweet_count<=10000].retweet_count.hist()
plt.title('Histogram of retweet count',fontsize=16,weight='bold')
plt.xlabel('Retweet_count',weight='bold')
plt.ylabel('Frequency',weight='bold')
plt.rcParams['figure.figsize']=(10,10)


# In[91]:


x=twitter_archive_master.retweet_count.mean()
mx=twitter_archive_master.retweet_count.max()

plt.bar([1,2],[x,mx],tick_label=[' mean retweet count','max retweet count'])
plt.rcParams['figure.figsize']=(10,10)


# In[92]:


twitter_archive_master.favorite_count.describe()


# In[96]:


twitter_archive_master[twitter_archive_master.favorite_count<=18000].favorite_count.hist()
plt.title('Histogram of favorite count',fontsize=16,weight='bold')
plt.xlabel('Favourite count',weight='bold')
plt.ylabel('Frequency',weight='bold')


# In[102]:


z=twitter_archive_master.favorite_count.max()
mnn=twitter_archive_master.favorite_count.mean()
plt.bar([1,2],[mnn,z],tick_label=[' mean favorite count','max favorite count'])
plt.rcParams['figure.figsize']=(10,10)


# In[162]:


df_max_fv=twitter_archive_master.groupby('dog_breed').max()[['favorite_count']]
df_max_fv.head(10).sort_values(by='favorite_count',ascending=False)


# In[163]:


df_max_rt=twitter_archive_master.groupby('dog_breed').max()[['retweet_count']]
df_max_rt.head(10).sort_values(by='retweet_count',ascending=False)


# In[89]:



twitter_archive_master.plot(x='timestamp',y='retweet_count',kind='line')
plt.title('Retweet counts over time',weight='bold')


# In[ ]:




