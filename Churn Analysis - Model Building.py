#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# #### Reading csv

# In[2]:


df=pd.read_csv(r"F:\NCPL\Project\Python\tel_churn.csv")
df.head()


# In[3]:


df=df.drop('Unnamed: 0',axis=1)


# In[4]:


x=df.drop('Churn',axis=1)
x


# In[5]:


y=df['Churn']
y


# ##### Train Test Split

# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# #### Decision Tree Classifier

# In[7]:


model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[8]:


model_dt.fit(x_train,y_train)


# In[9]:


y_pred=model_dt.predict(x_test)
y_pred


# In[12]:


model_dt.score(x_test,y_test)


# In[15]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[37]:


from imblearn.combine import SMOTEENN
sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(x,y)


# In[38]:


xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)


# In[39]:


model_dt_smote=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[42]:


model_dt_smote.fit(xr_train,yr_train)
yr_predict = model_dt_smote.predict(xr_test)
model_score_r = model_dt_smote.score(xr_test, yr_test)
print(model_score_r)
print(metrics.classification_report(yr_test, yr_predict))


# In[41]:


print(metrics.confusion_matrix(yr_test, yr_predict))


# ###### Now we can see quite better results, i.e. Accuracy: 92 %, and a very good recall, precision & f1 score for minority class.
# 
# ###### Let's try with some other classifier.

# #### Random Forest Classifier

# In[32]:


from sklearn.ensemble import RandomForestClassifier


# In[33]:


model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[34]:


model_rf.fit(x_train,y_train)


# In[28]:


y_pred=model_rf.predict(x_test)


# In[29]:


model_rf.score(x_test,y_test)


# In[30]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[ ]:





# In[ ]:





# In[44]:


sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(x,y)


# In[45]:


xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)


# In[46]:


model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[47]:


model_rf_smote.fit(xr_train1,yr_train1)


# In[51]:


yr_predict1 = model_rf_smote.predict(xr_test1)


# In[52]:


model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)


# In[53]:


print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))


# In[54]:


print(metrics.confusion_matrix(yr_test1, yr_predict1))


# #### Performing PCA

# In[55]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train1)
xr_test_pca = pca.transform(xr_test1)
explained_variance = pca.explained_variance_ratio_


# In[56]:


model=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[57]:


model.fit(xr_train_pca,yr_train1)


# In[58]:


yr_predict_pca = model.predict(xr_test_pca)


# In[59]:


model_score_r_pca = model.score(xr_test_pca, yr_test1)


# In[60]:


print(model_score_r_pca)
print(metrics.classification_report(yr_test1, yr_predict_pca))


# ##### With PCA, we couldn't see any better results, hence  finalising the model which was created by RF Classifier,

# ### Logistic Regression

# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


model_LR = LogisticRegression()


# In[66]:


model_LR.fit(x_train,y_train)


# In[67]:


y_pred=model_LR.predict(x_test)
y_pred


# In[68]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# ###### By this we can conclude With RF Classifier, also got  good results, infact better than Decision Tree.
# 

# 

# In[ ]:




