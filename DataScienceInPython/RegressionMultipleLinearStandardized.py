#Muliple Linear regression with Standardization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

##########
#Standardization: It is one kind of Normalization, Standardization is called Feature Scaling.
#                                 It is the process of transforming data into a standard scale.
# Formula:               standardized variable = (X - X_Mean)/Standard Deviation
#NOTE:                     in terms of STANDARDIZED variable     "intercept"     is called   "bias"
#                                                                                                           "coefficient"   is called  "weight"
universityData = pd.read_csv('C:/Users/HP 840 G1/Desktop/Semester 5/Artificial Intelligence/Lab Work/Regression/2.Multiple linear regression.csv')

#will print the top 5 data of the data frame 
print(universityData.head()) 

#Maping the categorial data to numerical data
universityData['Attendence'] = universityData['Attendence'].map({'Yes': 1, 'No': 0})

#will print the top 5 data of the data frame 
print(universityData.head()) 

#setting feature and target variable, it is supervised learning where input and output exist
x = universityData[['SAT','Attendence']] # ekhane 'x' already 2D 
y = universityData['GPA']

#StandardScaler(copy = True, with_mean = True, with_std = True) by default
#creating a object of StandardScaler Class
scaler = StandardScaler()

#fitting the x variable into scaler, mane holo je variable ta scale korbo seta bollam
scaler.fit(x)

#scaling the x variable
x_scaled = scaler.transform(x)

#printing the standardized variable x
#print(x_scaled)

#in order to use Sklearn we have to create the of that particular class
LRegre = LinearRegression()

#Creating Linear Regression using fit(x_scaled,y) 
LRegre.fit(x_scaled, y) # fit(input,output)

#printing the R squared value which show the goodness of the regerssion model
RSquared = LRegre.score(x_scaled, y)
print("R squared : ",RSquared) 

#printing the Adjusted R squared value which show the goodness of the regerssion model
n = x_scaled.shape[0]
print("Number of Observaitons:",n)
p = x_scaled.shape[1]
print("Number of Predictors:",p)
Adjusted_RSquared = 1-(1-RSquared)*((n-1)/(n-p-1))
print("Adjusted R Squared : ",Adjusted_RSquared) 
print("jehetu R2 > AdR2 so predictor barano te model er goodness bare ni")

#printing the intercept of the simple linear regression equation
intercept = LRegre.intercept_
print("Intercept : ",intercept) #ata only akta float because intercept akta e hoy regression e

#printing the intercept of the simple linear regression equation
coefficient = LRegre.coef_
print("Coefficients b1,b2 : ",coefficient) #it is list because co efficient akadhik hote pare like multiple L. R.

#akhon amra akta important term dekhbo na seta holo p-value karon standardized feature er kkhetre p-value differ kore na
fRegression, pValues = f_regression(x_scaled,y)
print("f-regression : ",fRegression)
print("p-values : ",pValues)
pValue_SAT = pValues[0]
pValue_Attendence = pValues[1]
print("p-value of SAT :",pValue_SAT.round(7),"     p-value of Attendence :",pValue_Attendence.round(7))

#Now we will  create a summary Table
Regression_summary = pd.DataFrame({'Features': ["Bias(intercept)","SAT","Attendence"]})
Regression_summary[ 'Weights(coefficients)'] = intercept.round(6),pValue_SAT.round(6),pValue_Attendence.round(7)
print(Regression_summary)

#predicting GPA using predict()
new_students = pd.DataFrame({'SAT': [1714,1664], 'Attendence': [0,0]})

#in new_students data frame, creating a new column 'Predicted GPA' with the predicted values
scaled_new_students = scaler.transform(new_students)

#printing the dataframe with the predicted GPA
#print(scaled_new_students)

#in new_students data frame, creating a new column 'Predicted GPA' with the predicted values
new_students['Predicted GPA'] = LRegre.predict(scaled_new_students)

#printing the dataframe with the predicted GPA
print(new_students)


##Now we will draw a plot of the regression
#plt.scatter(universityData['SAT'],y)
#ypredct =  0.275 + 0.001655*universityData['SAT'] # y = b0 + b1x
#plt.plot(universityData['SAT'],ypredct)
#plt.show()

##Statsmodel library amader k je information diyechilo
#Regression using OLS model
#xx = sm.add_constant(x_scaled) # x holo akta constand jetar value holo 1
#result = sm.OLS(y,xx).fit() # "y = b0x0+b1x1" jodi "x0 =1" hoy se khetrei "y = b0 + b1x1" hoy
#printing the OLS model where regression exists
#print(result.summary())