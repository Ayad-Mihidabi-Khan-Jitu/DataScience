###Simple Linear Regression

##importing necessary packages for simple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns # it makes the graph more beautiful
sns.set() #setting seaborn for all plots 

##loading table using pandas package 
#xl e jeta TABLE python e seta DATA FRAME
#SYNTEX :  pandas.read_csv("location/of/the/csv/file.csv")
datasetOfUniversity = pd.read_csv('C:/Users/HP 840 G1/Desktop/Semester 5/Artificial Intelligence/Lab Work/Regression/1.Simple linear regression.csv')
#here "datasetOfUniversity" is a data frame
#ekhane SAT er marks er opor base kore college e graduation er por GPA predict kore hocche

#printing the data frame / table 
print(datasetOfUniversity)

#important informantion of the data frame
print(datasetOfUniversity.describe())

#Udata=datasetOfUniversity.iloc[:,1:2].values
#print(Udata)

#Here dependent(y) variable is GPA 
#Here independent(x) variable is SAT
#we know, y = mx + c is a straight line where, c = intercept of axis y, m = slope
#so our predict value is y` = b0x0 + b1x1 => y` = b0 + b1x1 ; x0=1

#data frame theke 'GPA col' y e rakhlam 
y = datasetOfUniversity['GPA']
#data frame theke 'SAT col' x e rakhlam 
x1 = datasetOfUniversity['SAT']

#ploting a graph of the data frame using 'matplotlib.pyplot'
#plt.scatter(x1,y)
#plt.xlabel('SAT', fontsize=20)
#plt.ylabel('GPA', fontsize=20)
#plt.show()

#Regression(OLS-Ordinary Least Square) itself , we will get value of b0 and b1 from the model
#p value show the variability of the consecutive variable
#for const: p value less then 0.05 means it is significant in 95% cases it is not 0,
#for SAT: p value 0.00 means it is very perfect for predicting GPA
#t value is hypothesis test: beshi akta jani na aro deeply jante hobe, ekhane b0 is not zero

#It describes the over all model and shows important data
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary())

#ploting regression line of the data frame using 'matplotlib.pyplot'
plt.scatter(x1,y) 
predty = 0.0017 * x1 + 0.275  #GPA =  0.0017 * SAT + 0.275 
fig = plt.plot(x1,ypredt,lw=2,c='red',label ='regression line') #plot(x,y,linewitdh,color,lebel)
#fig = plt.plot(x1,ypredt)       #plot(x,y)
plt.show()

