#Scikit-learn in short sklearn is a MACHINE LEARNING package.
#It comes from the Toolkit called "SciPy Toolkit"
#Scikit is build on "Numpy Scipy and matplotlib"
#It is very fast, efficient and prefers working with arrays
#It is NUMERICALLY STABLE, very efficient in complex computation
#It provides : REGRESSION, CLASSIFICATION, CLUSTERING,
#                        SUPPORT VECTOR MACHINES, DIMENSIONALITY REDUCTION
#                        Infect it is very popular package for MACHINE LEARNING
#NOTE: in machine learning "input" is called "feature" and "output" is called "target"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

universityData = pd.read_csv('C:/Users/HP 840 G1/Desktop/Semester 5/Artificial Intelligence/Lab Work/Regression/1.Simple linear regression.csv')

#will print the top 5 data of the data frame 
print(universityData.head()) 

#setting feature and target variable, it is supervised learning where input and output exist
x = universityData['SAT']
y = universityData['GPA']

x_matrix = x.values.reshape(-1,1) # (observations,1) = (84,1) ata o valid, ata korbo karon skl mul linear regree kore
#print(x_matrix.shape)

#checking the size of x and y
print(x.shape)
print(y.shape)

#in order to use Sklearn we have to create the of that particular class
#the  LinearRegression(copy_X = True, fit_intercept = True, n_jobs = 1, normalize = false) aguli default
#here, normalization : regression ta normalized kore,  copy_X = Copying the data into the new variable
#           n_jobs : the number of CPU to complete the calculation, fit_intercept : the constant x0 = 1    
LRegre = LinearRegression()

#Creating Linear Regression using fit(x,y)
LRegre.fit(x_matrix, y) # fit(input,output)

#printing the R squared value
print("R squared : ",LRegre.score(x_matrix, y))

#printing the intercept of the simple linear regression equation
intercept = LRegre.intercept_
print("Intercept : ",intercept) #ata only akta float because intercept akta e hoy regression e

#printing the intercept of the simple linear regression equation
coefficient = LRegre.coef_
print("Coefficient b1 : ",coefficient) #it is list because co efficient akadhik hote pare like multiple L. R.

#predicting a y using predict()
new_students = pd.DataFrame(data = [1700,1800], columns = ['SAT']) # another process of creating DF

#printing the predicted GPA, Here predict is a function takes the input arguments or data frame as input
print(LRegre.predict(new_students))

#in new_students data frame, creating a new column 'Predicted GPA' with the predicted values
new_students['Predicted GPA'] = LRegre.predict(new_students)

#printing the dataframe
print(new_students)

#now we will draw a plot of the regression
plt.scatter(x,y)
ypredct = intercept + coefficient*x # y = b0 + b1x
plt.plot(x,ypredct)
plt.show()