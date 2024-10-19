#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[83]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score,roc_curve, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[4]:


df=pd.read_csv("/Users/prabhnoorsingh/Documents/churn.csv")


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


categorical_variables=[col for col in df.columns if df[col].dtype =="object" or df[col].nunique()<= 11
                       and col != "Exited"]
categorical_variables


# In[10]:


numeric_variables=[col for col in df.columns if df[col].dtype!="object" and df[col].nunique()>11 and col!="CustomerId"]
numeric_variables


# In[11]:


df["Exited"].value_counts()


# In[12]:


churn=df.loc[df["Exited"]==1]
churn.head()


# In[13]:


not_churn=df.loc[df["Exited"]==0]
not_churn.head()


# In[14]:


not_churn['Tenure'].value_counts().sort_values()


# In[15]:


churn['Tenure'].value_counts().sort_values()


# In[16]:


not_churn['NumOfProducts'].value_counts().sort_values()


# In[17]:


churn['NumOfProducts'].value_counts().sort_values()


# # Has Credit Card

# In[18]:


not_churn['HasCrCard'].value_counts()


# In[19]:


churn['HasCrCard'].value_counts()


# # Is active member

# In[20]:


not_churn['IsActiveMember'].value_counts()


# In[21]:


churn['IsActiveMember'].value_counts()


# # Geography
# 

# In[22]:


not_churn['Geography'].value_counts().sort_values()


# In[23]:


churn['Geography'].value_counts().sort_values()


# # Gender

# In[24]:


not_churn['Gender'].value_counts()


# In[25]:


churn['Gender'].value_counts()


# # Numrical Variables

# ## Credit Score

# In[26]:


not_churn['CreditScore'].describe()


# In[27]:


plt.figure(figsize=(8,6))
plt.xlabel("Credit Score")
plt.hist(not_churn['CreditScore'],bins=15,alpha=1.0,label='Not Churn')
plt.legend(loc="upper right")
plt.show()


# In[28]:


churn['CreditScore'].describe()


# In[29]:


plt.figure(figsize=(8,6))
plt.xlabel("churn")
plt.hist(churn['CreditScore'],bins=15,alpha=0.9,label="churn")
plt.legend()
plt.show()


# In[30]:


sns.catplot(x="Exited",y="CreditScore",data = df)


# # Age

# In[31]:


not_churn['Age'].describe()


# In[32]:


plt.figure(figsize=(8,6))
plt.xlabel("Age")
plt.hist(not_churn['Age'],bins=15,alpha=0.8,label="not churn")
plt.legend()
plt.show()


# In[33]:


churn['Age'].describe()


# In[34]:


plt.figure(figsize=(8,6))
plt.xlabel("Age")
plt.hist(churn['Age'],bins=15,alpha=0.8,label="churn")
plt.legend()
plt.show()


# In[35]:


sns.catplot(x="Exited",y="Age",data=df)


# # Balance

# In[36]:


not_churn['Balance'].describe()


# In[37]:


plt.figure(figsize=(8,6))
plt.xlabel("balance")
plt.hist(not_churn['Balance'],bins=15,alpha=0.8,label="not churn")
plt.legend()
plt.show()


# In[38]:


churn['Balance'].describe()


# In[39]:


plt.figure(figsize=(8,6))
plt.xlabel("balance")
plt.hist(churn['Balance'],bins=15,alpha=0.8,label="churn")
plt.legend()
plt.show()


# In[40]:


not_churn['EstimatedSalary'].describe()


# In[41]:


plt.figure(figsize=(8,6))
plt.xlabel("Estimated Salary")
plt.hist(not_churn['EstimatedSalary'],bins=15,alpha=0.8,label="not churn")
plt.legend()
plt.show()


# In[42]:


churn['EstimatedSalary'].describe()


# In[43]:


plt.figure(figsize=(8,6))
plt.xlabel("Estimated Salary")
plt.hist(churn['EstimatedSalary'],bins=15,alpha=0.8,label="churn")
plt.legend()
plt.show()


# In[44]:


#correlation matrix
numeric_df=df.select_dtypes(include=[float,int])
k=10
cols =numeric_df.corr().nlargest(k,'Exited')['Exited'].index
cm=numeric_df[cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,cmap='viridis')
plt.show()


# In[45]:


def outlier_threshold(dataframe,variable,low_quantile=0.05,up_quantile=0.95):
    quantile_one=dataframe[variable].quantile(low_quantile)
    quantile_three=dataframe[variable].quantile(up_quantile)
    interquantile_range=quantile_three-quantile_one
    up_limit=quantile_three+ 1.5 * interquantile_range
    low_limit=quantile_one- 1.5 * interquantile_range
    return low_limit,up_limit
    


# In[46]:


def has_outliers(dataframe,numeric_cols):
#     variable_names=[]
    for col in numeric_cols:
        low_limit,up_limit=outlier_threshold(dataframe,col)
        
        if dataframe[(dataframe[col]>up_limit) | (dataframe[col]<low_limit)].any(axis=None):
            
            number_of_outliers=dataframe[(dataframe[col]>up_limit) | (dataframe[col]<low_limit)].shape[0]
            print(col,":",number_of_outliers,"outliers")
        
        
#         if outliers.shape[0]>0:
#             print(f"{col}:{outlier.shape[0]} outliers")


# In[47]:


for var in numeric_variables:
    print(var,"has ",has_outliers(df,[var]),"Outliers")


# ## Feature Engineering

# In[48]:


df["NewTenure"]=df["Tenure"]/df["Age"]
df["NewCreditScore"]=pd.qcut(df["CreditScore"],6,labels=[1,2,3,4,5,6])
df["NewAgeScore"]=pd.qcut(df['Age'],8,labels=[1,2,3,4,5,6,7,8])
df["NewBalanceScore"]=pd.qcut(df['Balance'].rank(method="first"),5,labels=[1,2,3,4,5])
df["NewEstSalaryScore"]=pd.qcut(df['EstimatedSalary'],10,labels=[1,2,3,4,5,6,7,8,9,10])


# In[49]:


df.head()


# ## One Hot Encoding

# In[50]:


list=["Gender","Geography"]
df=pd.get_dummies(df,columns=list,drop_first=False)
df.head()


# In[51]:


df=df.drop(["CustomerId","Surname"],axis=1)


# In[52]:


df.head()


# ## Scaling

# In[53]:


def robust_scaler(variable):
    var_median=variable.median()
    quantile1=variable.quantile(0.25)
    quantile3=variable.quantile(0.75)
    interquantile_range=quantile3-quantile1
    if int(interquantile_range)==0:
        quantile1=variable.quantile(0.05)
        quantile3=variable.quantile(0.95)
        interquantile_range=quantile3-quantile1
        if int(interquantile_range)==0:
            quantile1=variable.quantile(0.01)
            quantile3=variable.quantile(0.99)
            interquantile_range=quantile3-quantile1
            z= (variable-var_median)/interquantile_range
            return round(z,3)
        z= (variable-var_median)/interquantile_range
        return round(z,3)
    else:
        z=(variable-var_median)/interquantile_range
        return round(z,3)


# In[54]:


new_cols_ohe=["Gender_Male","Gender_Female","Geography_Spain","Geography_Germany","Geography_France"]
like_num=[col for col in df.columns if df[col].dtype!="object" and len(df[col].value_counts())<=10]
cols_need_scale=[col for col in df.columns if col not in new_cols_ohe
                and col not in "Exited"
                and col not in like_num]
for col in cols_need_scale:
    df[col]=robust_scaler(df[col])
    


# In[55]:


df=df.drop("RowNumber",axis=1)


# In[56]:


df.head()


# ## Modeling

# In[60]:


X=df.drop("Exited",axis=1)
y=df["Exited"]


# In[71]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=12345)

models=[('LR', LogisticRegression(random_state=12345)),
       ('KNN',KNeighborsClassifier()),
       ('CART',DecisionTreeClassifier(random_state=12345)),
       ('RF',RandomForestClassifier(random_state=12345)),
       ('SVR',SVC(gamma='auto',random_state=12345)),
       ('GB',GradientBoostingClassifier(random_state=12345))
       #('LGBM',LGBGMClassifier(random_state=12345) 
        ]
results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=10)
    cv_results=cross_val_score(model,X,y,cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)"%(name, cv_results.mean(),cv_results.std())
    print(msg)


# In[72]:


model_GB=GradientBoostingClassifier(random_state=12345)
model_GB.fit(X_train,y_train)
y_pred=model_GB.predict(X_test)
conf_mat=confusion_matrix(y_pred,y_test)
print(conf_mat)


# In[75]:


print(classification_report(model_GB.predict(X_test),y_test))


# In[84]:


lgb_model=LGBMClassifier()
#model tuning
lgbm_params={
    'colsample_bytree':0.5,
    'learning_rate':0.01,
    'max_depth':6,
    'n_estimators':500
}
lgbm_tuned=LGBMClassifier(**lgbm_params).fit(X,y)


# In[94]:


gbm_model=GradientBoostingClassifier()
gbm_params={
    'learning_rate':0.01,
    'n_estimators':500,
    'max_depth':6,
    'subsample':1
}
gbm_tuned=GradientBoostingClassifier(**gbm_params).fit(X,y)


# In[96]:


models=[("LightGBM",lgbm_tuned),
       ("GB",gbm_tuned)]
results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=10)
    cv_score=cross_val_score(model,X,y,cv=10,scoring='accuracy')
    results.append(cv_score)
    names.append(name)
    msg= " %s : %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    


# In[103]:


for name,model in models:
    base=model.fit(X_train,y_train)
    pred=model.predict(X_test)
    acc_score=accuracy_score(y_test,pred)
    print(pred)
    print(acc_score*100)\
    


# In[ ]:




