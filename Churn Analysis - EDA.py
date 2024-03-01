#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Analysis

# In[2]:


#import the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


telco_base_data = pd.read_csv(r'F:\NCPL\Project\Python\WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[4]:


telco_base_data.head()


#  rows and cols

# In[5]:


telco_base_data.shape


# In[6]:


telco_base_data.columns.values


# In[7]:


# Checking the data types of all the columns
telco_base_data.dtypes


# In[8]:


#  Descriptive statistics of numeric variables
telco_base_data.describe()


# 75% of customers have tenure of 55 months
# 
# Average Monthly charges are USD 64.76 whereas 75% customers pay  USD 89.85 per month

# In[9]:


telco_base_data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count")
plt.ylabel("Target Variable")
plt.title("Count of TARGET Variable per category");


# In[10]:


100*telco_base_data['Churn'].value_counts()/len(telco_base_data['Churn'])


# In[11]:


telco_base_data['Churn'].value_counts()


#  Data is highly imbalanced, ratio = 73:27
#  
#  So analysing the data with other features while taking the target values to get some insights.

# In[12]:


# Summary to check null values
telco_base_data.info()   


# ## Data Cleaning
# 

# **1.** copying  of base data for manupulation & processing

# In[13]:


telco_data = telco_base_data.copy()


# **2.** Total Charges should be numeric amount so converting it to numerical data type

# In[14]:


telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.isnull().sum()


# **3.** There are 11 missing values in TotalCharges column.

# In[15]:


telco_data.loc[telco_data ['TotalCharges'].isnull() == True]


# In[16]:


#Removing missing values 
telco_data.dropna(how = 'any', inplace = True)

#telco_data.fillna(0)


# **5.** Divide customers into bins based on tenure

# In[17]:


#  maximum tenure
print(telco_data['tenure'].max()) 


# In[18]:


#  tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)


# In[19]:


telco_data['tenure_group'].value_counts()


# **6.** Removed columns which are not  not required 

# In[20]:


#drop column customerID and tenure
telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
telco_data.head()


# ## Data Exploration
#  Plot distibution of individual predictors by churn

# ### Univariate Analysis

# In[21]:


for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor, hue='Churn')


#  Convering the target variable 'Churn'  in a binary numeric variable i.e. Yes=1 ; No = 0

# In[22]:


telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)


# In[23]:


telco_data.head()


# **3.** Converting all the categorical variables into dummy variables

# In[24]:


telco_data_dummies = pd.get_dummies(telco_data)    # One Hot Encoding
telco_data_dummies.head()


#  Relationship between Monthly Charges and Total Charges

# In[25]:


sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges')


# Total Charges increase as Monthly Charges increase

#  corelation of all predictors with 'Churn'

# In[26]:


plt.figure(figsize=(20,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


#  Insights:
# 
# **HIGH** Churn seen in case of  **Month to month contracts**, **No online security**, **No Tech support**, **First year of subscription** and **Fibre Optics Internet**
# 
# **LOW** Churn is seens in case of **Long term contracts**, **Subscriptions without internet service** and **The customers engaged for 5+ years**
# 
# 
# 
# 

# In[27]:


plt.figure(figsize=(12,12))
sns.heatmap(telco_data_dummies.corr(),cmap="Paired")


# 

# ### Bivariate Analysis

# In[28]:


new_df1_target0=telco_data.loc[telco_data["Churn"]==0]    # Non churners
new_df1_target1=telco_data.loc[telco_data["Churn"]==1]     # Churners


# In[29]:


def uniplot(df,col,title,hue =None):  
    plt.title(title) 
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue)
    plt.show()


# In[30]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[31]:


uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[32]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[33]:


uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[34]:


uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')


# In[35]:


uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# # CONCLUSION

# 
# 
# 1. Electronic check medium are the highest churners
# 2. Contract Type - Monthly customers are more likely to churn because of no contract terms, as they are free to go customers.
# 3. No Online security, No Tech Support category are high churners
# 4. Non senior Citizens are high churners
# 

# In[37]:


telco_data_dummies.to_csv(r'F:\NCPL\Project\Python\tel_churn.csv')


# In[ ]:




