#Muliple Linear regression with determining the p-value of the inputs/features/predictors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

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

#in order to use Sklearn we have to create the of that particular class
LRegre = LinearRegression()

#Creating Linear Regression using fit(x,y) # ekhane x already two dimensional so reshape korar korkar nai
LRegre.fit(x, y) # fit(input,output)

#printing the R squared value which show the goodness of the regerssion model
RSquared = LRegre.score(x, y)
print("R squared : ",RSquared) 

#amra jani R squared value ata simple linear regression er jonno
#multiple linear regression er jonno holo "adjusted R squared"
#Sklearn adjusted R squared er jonno kono function provide kore na
#So ata amader jonno rekhe dise : adjusted R squared = {1-(1-R2)}*{(n-1)/(n-p-1)}
#here, R2 = R squared,  n = number of observations,  p = number of inputs (independent variable or number of predictors) 
n = x.shape[0]
print("Number of Observaitons:",n)
p = x.shape[1]
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

#predicting a y using predict()
#new_students = pd.DataFrame(data = {[1700,1800],[1,0]}, columns = {['SAT'],['Attendence']}) # another process of creating DF
new_students = pd.DataFrame({'SAT': [1700,1800], 'Attendence': [1,0]})

#in new_students data frame, creating a new column 'Predicted GPA' with the predicted values
new_students['Predicted GPA'] = LRegre.predict(new_students)

#printing the dataframe with the predicted GPA
print(new_students)

#akhon amra akta important term dekhbo seta holo p-value
#regression model e p-value expresses the importance of the predictor to predict(needed or unneeded)
#jar p-value joto kom se toto beshi important,   0.5 er choto hole seta rakha jay otherwise discard it
#################################################################
#je term tar maddhome sklearn ata kore setar nam holo feature selection, 
#feature selection ta mutolo f-regressoin er maddhome hoy, [sklearn.feature_selection import f_regression]
#akhon kotha holo f-regression ta ki??
#f-regression creates a simple linear regression of each feature and dependent variable
#amader example e:        "SAT"                   ->          "GPA"            produces "p-value"
#                                             "Attendence"    ->         "GPA"             produces "p-value"

#creating f-regression using f_regression(feature,target) 
#it returns n lists depending on the nmbr of features whr in each list thr are two lists [f-regressions],[p-valus]
fRegression, pValues = f_regression(x,y)
print("f-regression : ",fRegression)
print("p-values : ",pValues)
pValue_SAT = pValues[0]
pValue_Attendence = pValues[1]
print("p-value of SAT :",pValue_SAT.round(),"     p-value of Attendence :",pValue_Attendence.round())

#Now we will  create a summary Table
Regression_summary = pd.DataFrame()
Regression_summary['Features'] = x.columns.values
Regression_summary['Co-efficients'] = coefficient
Regression_summary['p-values'] = pValues.round(3)
print(Regression_summary)

#now we will draw a plot of the regression
plt.scatter(universityData['SAT'],y)
ypredct =  0.275 + 0.001655*universityData['SAT'] # y = b0 + b1x
plt.plot(universityData['SAT'],ypredct)
plt.show()

##Statsmodel library amader k je information diyechilo
#Regression using OLS model
#xx = sm.add_constant(x) # x holo akta constand jetar value holo 1
#result = sm.OLS(y,xx).fit() # "y = b0x0+b1x1" jodi "x0 =1" hoy se khetrei "y = b0 + b1x1" hoy
#printing the OLS model where regression exists
#print(result.summary())