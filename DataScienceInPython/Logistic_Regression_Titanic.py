import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
##import data
pd.set_option('display.max_columns', None) 
titanic_data = pd.read_csv("Titanic.csv")
#print('Table \n {}'.format(titanic_data.head(10)))
#print('No. of passenger {}'.format(len(titanic_data)))

##analyze data 
#sns.countplot(x='Survived',data=titanic_data)
#sns.countplot(x='Survived',hue='Sex',data=titanic_data)
#sns.countplot(x='Survived',hue='Pclass',data=titanic_data)
#sns.countplot(x='Survived',hue='Age',data=titanic_data)

#titanic_data['Age'].plot.hist()
#titanic_data['Fare'].plot.hist(bins=20,figsize=(10,5))
#sns.countplot(x='SibSp',data=titanic_data)
#print(titanic_data.info())
#plt.show()

##Data Wrangling
#print(titanic_data.isnull())
print(titanic_data.isnull().sum())
#print("nulls: $",titanic_data.isnull().sum())
#sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='Accent')
#sns.
#plt.show()
