###Multiple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

#loading table into a data frame
UniversityData_Raw = pd.read_csv('C:/Users/HP 840 G1/Desktop/Semester 5/Artificial Intelligence/Lab Work/Regression/2.Multiple linear regression.csv')

#printing the data frame
print(UniversityData_Raw)

#Copying the data into the new variable
UniversityData = UniversityData_Raw.copy()

#Making the Attendence into a dummy vairabe so that we can use it to Regression
#Categorical data to numerical conversion
UniversityData['Attendence'] = UniversityData['Attendence'].map({'Yes':1,'No':0}) 

#printing the data frame where 'Attendence' Yes = 1 and No = 0
print(UniversityData)

#printing important data of UniversityData
print(UniversityData.describe())

#Independent and dependent variable setting
y = UniversityData['GPA']
x1 = UniversityData[['SAT','Attendence']]

#Regression using OLS model
x = sm.add_constant(x1) # x holo akta constand jetar value holo 1
result = sm.OLS(y,x).fit() # "y = b0x0+b1x1" jodi "x0 =1" hoy se khetrei "y = b0 + b1x1" hoy

#x holo akta data frame jekhane const = 1, 'SAT', 'Attendence' er values
print(x)

#printing the OLS model where regression exists
print(result.summary())

# From the regresssion we get: b0 = 1.623,           b1 = 0.0008,                       b2 = 0.3108
#           mane holo :            intercept = 1.623,         SAT = 0.0008,     Attendence = 0.3108   
# therefore the multiple linear regression is:     y = 1.623 + 0.0008x1 + 0.3108x2
# mane holo :  GPA = 1.623 + 0.0008SAT + 0.3108Attendence

#GPA = 1.623 + 0.0008SAT   where Attendence = 0
#GPA = 1.933 + 0.0008SAT   where Attendence = 1

#scatter graph e SAT vs GPA dekhbo 
plt.scatter(UniversityData['SAT'],y,c=UniversityData['Attendence'],cmap='RdYlGn_r')

#GPA = 1.623 + 0.0008SAT   where   Attendence = 0
predctY_no = 1.623 + 0.0008*UniversityData['SAT']
plt.plot(UniversityData['SAT'], predctY_no, c='Green', label='Absent')

#GPA = 1.623 + 0.0008SAT   where   Attendence = 1
predctY_yes = 1.933 + 0.0008*UniversityData['SAT']
plt.plot(UniversityData['SAT'], predctY_yes, c='Red', label='Present')

#GPA =  0.275 + 0.0017SAT   where   Attendence kono fact na
predctY = 0.0017 * UniversityData['SAT'] + 0.275  #GPA =  0.0017 * SAT + 0.275 
plt.plot(UniversityData['SAT'], predctY, c='Black', label='Attendence Not Considered')

plt.title('GPA Graph')
plt.xlabel('SAT',fontsize = 20)
plt.ylabel('GPA',fontsize = 20)
plt.legend()
plt.show()

##akhon akta duijon manus k niye tader GPA predict korbo oporer Regresion er opor base kore
#creating a table which is called data frame in python:
#the table as follows
#   const     SAT     Attendence
#   1            1700           0
#   1            1670           1
new_students = pd.DataFrame({'const': 1, 'SAT' : [1700,1670], 'Attendence': [0,1]}) #creating DF new_students
print(new_students)

#renaming the indices 0 and 1 into jitu and hasan
new_students = new_students.rename(index={0: 'jitu',1: 'hasan'}) 
print(new_students)

#result holo amader sei model (Multiple Linear Regression) jar opor base kore predict korbo
#predict() function ta argument hesebe result toiri korte ja ja legeche (const,'SAT','Attendence')
predictedGPAs = result.predict(new_students) # new_students ta te required arguments ache
print(predictedGPAs)

#predicted GPA list diye akta data frame create korchi jate seta new_student data frame er join kora jay
DFofPredictedGPA = pd.DataFrame({'PredictedGPA': predictedGPAs})
new_students_with_prediectdGPA = new_students.join(DFofPredictedGPA)
print(new_students_with_prediectdGPA)

#jehetu constant column ta dorkar nai tai delete korchi 
del new_students_with_prediectdGPA['const']
print(new_students_with_prediectdGPA)