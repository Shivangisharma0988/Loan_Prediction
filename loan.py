#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading the libraries
import pandas as pd


# In[2]:


import numpy as np
import seaborn as sns


# In[3]:


import matplotlib.pyplot as plt
#below condition is necessary for jupyter to use functions of matplotlib.
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[4]:


train=pd.read_csv("train.csv")


# In[5]:


train=pd.read_csv("train.csv")


# In[6]:


test=pd.read_csv("test.csv")


# In[7]:


train_original=train.copy() 
test_original=test.copy()


# In[8]:


train.columns


# In[9]:


test.columns


# In[10]:


train.dtypes


# In[11]:


test.dtypes


# In[12]:


train.shape


# In[13]:


test.shape


# In[14]:


train['Loan_Status'].value_counts()


# In[15]:


train['Loan_Status'].value_counts()


# In[16]:


train['Loan_Status'].value_counts(normalize=True)


# In[17]:


train['Loan_Status'].value_counts(normalize=True)


# In[18]:


train['Loan_Status'].value_counts()


# In[19]:


train['Loan_Status'].value_counts().plot.bar()


# In[20]:


train


# In[21]:


plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(title='gender')


# In[22]:


train['Gender'].value_counts(normalize=True).plot.bar(figsize=(10,5), title= 'Gender')


# In[26]:


train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()


# In[27]:


plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar( figsize=(20,10),title= 'Gender') 
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show();


# In[28]:


sns.distplot(train['ApplicantIncome']);


# In[29]:


train['ApplicantIncome'].plot.box()


# In[30]:


train.boxplot(column='ApplicantIncome', by = 'Education') 
plt.suptitle("")


# In[31]:


gd=pd.crosstab(train['Gender'],train['Loan_Status'])
gd.div(gd.sum(1).astype(float),axis=0).plot(kind="bar",stacked='True',figsize=(4,4))


# In[32]:


train


# In[33]:


ed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
ed.div(ed.sum(1).astype(float),axis=0).plot(kind="bar",stacked='True',figsize=(4,4))


# In[23]:


ed=pd.crosstab(train['Dependents'],train['Loan_Status'])
ed.div(ed.sum(1).astype(float),axis=0).plot(kind="bar",stacked='True',figsize=(4,4))


# In[24]:


ed=pd.crosstab(train['Education'],train['Loan_Status'])
ed.div(ed.sum(1).astype(float),axis=0).plot(kind="bar",stacked='True',figsize=(4,4))


# In[25]:


ed=pd.crosstab(train['Credit_History'],train['Loan_Status'])
ed.div(ed.sum(1).astype(float),axis=0).plot(kind="bar",stacked='True',figsize=(4,4))


# In[26]:


ed=pd.crosstab(train['Property_Area'],train['Loan_Status'])
ed.div(ed.sum(1).astype(float),axis=0).plot(kind="bar",stacked='True',figsize=(4,4))


# In[27]:


Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 


# In[28]:


Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 


# In[29]:


Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show()


# In[30]:


#We will try to find the mean income of people for which the loan has been approved
#vs the mean income of people for which the loan has not been approved.

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[31]:


bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)


# In[32]:


Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')


# In[33]:



'''It shows that if coapplicant’s income is less the chances of loan approval are high. 
But this does not look right. The possible reason behind this may be that most of the applicants don’t have
any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it
So we can make a new variable in which we will combine the applicant’s 
and coapplicant’s income to visualize the combined effect of income on loan approval.'''


# In[34]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


# In[35]:


'''We can see that Proportion of loans getting approved for applicants having low
Total_Income is very less as compared to that of applicants with Average,
High and Very High Income.'''


# In[36]:


bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(test['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')


# In[37]:


train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)


# In[1]:


A=['1','2','3']
print(A)


# In[4]:


A=['1','2','3']

for a in A:

    print(2*a)


# In[ ]:





# In[ ]:





# In[ ]:




