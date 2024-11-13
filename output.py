import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix

cancer=pd.read_csv('data.csv')

cancer.head()

cancer.shape

cancer.tail()

cancer.isna().sum()

cancer.describe()

cancer.info()

cancer['diagnosis'].value_counts()

cancer['diagnosis'].value_counts().plot(kind='bar')

from sklearn.preprocessing import LabelEncoder

# Create a sample dataframe with categorical data
diagnosiss = pd.DataFrame({'diagnosis': ['M', 'B']})

print(f"Before Encoding the Data:\n\n{diagnosiss}\n")

# Create a LabelEncoder object
le = LabelEncoder()

# Fit and transform the categorical data
cancer['diagnosis'] = le.fit_transform(cancer['diagnosis'])

diagnosiss

cancer.describe()

cancer.describe().T

plt.figure(figsize=(25,25))
sns.heatmap(cancer.corr(),annot=True)

cancer.corr()


# Calculate the correlation matrix
corr_matrix = cancer.corr()

# Get the correlation values for the 'diagnosis' column
corr_with_diagnosis = corr_matrix['diagnosis'].abs().sort_values(ascending=False)
corr_with_diagnosis

x=cancer.drop(['diagnosis','id','Unnamed: 32'],axis=1)

y=cancer['diagnosis']

x

y

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train

x_test

y_train

y_test

s=StandardScaler()

x_train=s.fit_transform(x_train)
x_test=s.fit_transform(x_test)

model=LogisticRegression()

model.fit(x_train,y_train)

predict_test=model.predict(x_test)

predict_test

accuracy_score(y_test,predict_test)

predict_train=model.predict(x_train)

predict_train

accuracy_score(y_train,predict_train)



