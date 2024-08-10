#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[48]:


#Load Dataset
df = pd.read_csv('marketing_campaign.csv',sep = ';')


# In[5]:


df


# AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise
# Complain - 1 if customer complained in the last 2 years
# DtCustomer - date of customer’s enrolment with the company
# Education - customer’s level of education
# Marital - customer’s marital status
# Kidhome - number of small children in customer’s household
# Teenhome - number of teenagers in customer’s household
# Income - customer’s yearly household income
# MntFishProducts - amount spent on fish products in the last 2 years
# MntMeatProducts - amount spent on meat products in the last 2 years
# MntFruits - amount spent on fruits products in the last 2 years
# MntSweetProducts - amount spent on sweet products in the last 2 years
# MntWines - amount spent on wine products in the last 2 years
# MntGoldProds - amount spent on gold products in the last 2 years
# NumDealsPurchases - number of purchases made with discount
# NumCatalogPurchases - number of purchases made using catalogue
# NumStorePurchases - number of purchases made directly in stores
# NumWebPurchases - number of purchases made through company’s web site
# NumWebVisitsMonth - number of visits to company’s web site in the last month
# Recency - number of days since the last purchase

# In[8]:


print(df.info())


# In[18]:


df.columns


# In[9]:


df.head()


# In[23]:


df.describe(include = 'all')


# In[24]:


# clean up column names that contain whitespace
df.columns = df.columns.str.replace(' ', '')


# In[26]:


df.isnull().sum().sort_values(ascending = False)


# In[46]:


plt.figure(figsize=(8,4))
sns.distplot(df['Income'], kde=False, hist=True)
plt.title('Income distribution', size=16)
plt.ylabel('count');


# In[50]:


df['Income'].plot(kind='box', figsize=(3,4), patch_artist=True)


# In[ ]:


df['Income'] = df['Income'].fillna(df['Income'].median())


# In[54]:


# select columns to plot
df_to_plot = df.drop(columns=['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']).select_dtypes(include=np.number)

# subplots
df_to_plot.plot(subplots=True, layout=(5,5), kind='box', figsize=(12,14), patch_artist=True)
plt.subplots_adjust(wspace=0.5);


# In[55]:



df = df[df['Year_Birth'] > 1900].reset_index(drop=True)

plt.figure(figsize=(3,4))
df['Year_Birth'].plot(kind='box', patch_artist=True);


# In[56]:


df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])


# In[57]:


# Dependents
df['Dependents'] = df['Kidhome'] + df['Teenhome']

# Year becoming a Customer
df['Year_Customer'] = pd.DatetimeIndex(df['Dt_Customer']).year

# Total Amount Spent
mnt_cols = [col for col in df.columns if 'Mnt' in col]
df['TotalMnt'] = df[mnt_cols].sum(axis=1)

# Total Purchases
purchases_cols = [col for col in df.columns if 'Purchases' in col]
df['TotalPurchases'] = df[purchases_cols].sum(axis=1)

# Total Campaigns Accepted
campaigns_cols = [col for col in df.columns if 'Cmp' in col] + ['Response'] # 'Response' is for the latest campaign
df['TotalCampaignsAcc'] = df[campaigns_cols].sum(axis=1)

# view new features, by customer ID
df[['ID', 'Dependents', 'Year_Customer', 'TotalMnt', 'TotalPurchases', 'TotalCampaignsAcc']].head()


# In[65]:


# calculate correlation matrix
## using non-parametric test of correlation (kendall), since some features are binary
corrs = df.drop(columns='ID').select_dtypes(include=np.number).corr(method = 'kendall')

# plot clustered heatmap of correlations
sns.clustermap(corrs, cbar_pos=(-0.05, 0.8, 0.05, 0.18), cmap='coolwarm', center=0);


# In[66]:


sns.lmplot(x='Income', y='TotalMnt', data=df[df['Income'] < 200000]);


# In[69]:


plt.figure(figsize=(4,4))
sns.boxplot(x='Dependents', y='TotalMnt', data=df);#Plot illustrating positive effect of having dependents (kids & teens) on number of deals purchased:
#Plot illustrating negative effect of having dependents (kids & teens) on spending:


# In[70]:


plt.figure(figsize=(4,4))
sns.boxplot(x='Dependents', y='NumDealsPurchases', data=df)
#8plot illustrating positive effect of having dependents (kids & teens) on number of deals purchased:


# In[75]:


plt.figure(figsize=(4,4))
sns.boxplot(x='TotalCampaignsAcc',y='Income',data=df[df['Income'] < 200000])


# In[76]:


sns.lmplot(x='NumWebVisitsMonth', y='NumDealsPurchases', data=df);


# In[80]:


[df['AcceptedCmp2']].count(1)            


# In[ ]:




