##a complete project on regression.
##project e ja ja ache
#Data Cleanning : Atar jonno amader descriptive statistics er help nite hoy
#1. akta raw data theke kivebe data preprocess korte hoy
#2. kivabe features select korte hoy

#Creating Model



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()
pd.set_option('max_columns',None)
pd.set_option('max_rows',None)
pd.set_option('precision', 3)
pd.set_option('max_colwidth', 20)

##importing raw dataset
raw_DatasetOfCar = pd.read_csv('C:/Users/HP 840 G1/Desktop/Semester 5/Artificial Intelligence/Lab Work/Regression/3.Real-life example.csv')

#printing the first 5 record
print("The Raw dataset")
print(raw_DatasetOfCar.head(6))
print()

#printing the descriptive ststs of the raw dataset
print("The descriptive Information of the raw dataset")
desStatsOfDataset = raw_DatasetOfCar.describe(include = 'all')
print(desStatsOfDataset)
print()

###Cleaning dataset starts

##1. Deleting the column 'model' jehetu ata onk variety contain kore
#NOTE : axis = 0 means => row            |           axis = 1 means => column
cleaned_DatasetOfCar = raw_DatasetOfCar.drop(['Model'],axis = 1)

##2. Deleting the rows where nunnotomo akta cell o empty ache
print(cleaned_DatasetOfCar.isnull().sum())
cleaned_DatasetOfCar = cleaned_DatasetOfCar.dropna(axis = 0)

##3. Preprocessing Numerial Data
##Checking PDA making the plot to do as much as normalize
#"numerical data" that is "continuous data" gulir jonno mainly ata kore jate regression ta smooth hoy
#sns.displot(cleaned_DatasetOfCar['Price'],kde=True)
#plt.show()

#price k normalize kortesi 
q = cleaned_DatasetOfCar['Price'].quantile(0.99)
cleaned_DatasetOfCar = cleaned_DatasetOfCar[cleaned_DatasetOfCar['Price']<q]

#sns.displot(cleaned_DatasetOfCar['Price'],kde=True)
#plt.show()

#Mileage k normalize kortesi 
q = cleaned_DatasetOfCar['Mileage'].quantile(0.99)
cleaned_DatasetOfCar = cleaned_DatasetOfCar[cleaned_DatasetOfCar['Mileage']<q]

#sns.displot(cleaned_DatasetOfCar['Mileage'],kde=True)
#plt.show()

#EngineV k normalize kortesi 
q = 6.25 #jehetu amra jani engine volume always 6.25 er kom hoy [Collect from Wiki] 
cleaned_DatasetOfCar = cleaned_DatasetOfCar[cleaned_DatasetOfCar['EngineV']<q]

#sns.displot(cleaned_DatasetOfCar['EngineV'],kde=True)
#plt.show()

#Year k normalize kortesi 
q = cleaned_DatasetOfCar['Year'].quantile(0.01) 
cleaned_DatasetOfCar = cleaned_DatasetOfCar[cleaned_DatasetOfCar['Year']>q]

#sns.displot(cleaned_DatasetOfCar['Year'],kde=True)
#plt.show()

#Finally we cleaned the raw dataset
print("Cleaned dataset") 
print(cleaned_DatasetOfCar.describe(include = 'all'))

##jehetu amra price predict korbo tai price er shapekkhe onno variable gulir somporko dekhte hobe
#checking the OLS assumptions

#f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
#ax1.scatter(cleaned_DatasetOfCar['Year'],cleaned_DatasetOfCar['Price'])
#ax1.set_title("Price vs Year")
#ax2.scatter(cleaned_DatasetOfCar['Mileage'],cleaned_DatasetOfCar['Price'])
#ax2.set_title("Price vs Mileage")
#ax3.scatter(cleaned_DatasetOfCar['EngineV'],cleaned_DatasetOfCar['Price'])
#ax3.set_title("Price vs EngineV")
#plt.show()

##plot theke eta sposto je price er sathe Year,Mileage and EngineV er kono linear somporko nei
#borong price er sathe exponential somporko ache amra jani log transform use kore ata linear korajay
log_price = np.log(cleaned_DatasetOfCar['Price'])
cleaned_DatasetOfCar['Log_Price'] = log_price

#f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
#ax1.scatter(cleaned_DatasetOfCar['Year'],cleaned_DatasetOfCar['Log_Price'])
#ax1.set_title("Log_Price vs Year")
#ax2.scatter(cleaned_DatasetOfCar['Mileage'],cleaned_DatasetOfCar['Log_Price'])
#ax2.set_title("Log_Price vs Mileage")
#ax3.scatter(cleaned_DatasetOfCar['EngineV'],cleaned_DatasetOfCar['Log_Price'])
#ax3.set_title("Log_Price vs EngineV")
#plt.show()

#jehetu normal price kono kaj e asbe na tai Column 'Price' ta delete kore dilam
cleaned_DatasetOfCar = cleaned_DatasetOfCar.drop(['Price'],axis = 1)

##Year, Mieage and EngineV ara holo independent variable for predicting price
#predict korar age independent variable gulir moddle multiColinearity check korte hobe
#jodi multiCollinearilty thake taile seta remove kore felte hobe independent variable theke

#importing necessary packages
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Dataset theke independent variable guli alada korchi
IndeVariables = cleaned_DatasetOfCar[['Mileage','Year','EngineV']]
#MultiCollinearity dekhar jonno akta table vif
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(IndeVariables.values,i) for i in range(IndeVariables.shape[1])]
vif['Features'] = IndeVariables.columns
print(vif)

##MultiCollinearity checking 
# vif = 1 ; no multicollinearity
# 1<vif<5 ; perfectly okey
#10<vif ; unacceptable

#Table theke amra vif er value guli lokkho kori Year = 10.35
#Year variable ta Mileage and EnginV er sathe collinear
#So we will remove the Column 'Year'

cleaned_DatasetOfCar = cleaned_DatasetOfCar.drop(['Year'],axis=1)
#print(cleaned_DatasetOfCar.head())

##4. Preprocessing Categorial Data
#SYNETX of getting dummies : pd.get_dummies(df ,drop_first)
cleaned_DatasetOfCar_withDummies = pd.get_dummies(cleaned_DatasetOfCar,drop_first=True)
print(cleaned_DatasetOfCar_withDummies.head())

#Rearraging the dataset such as dependent variable sobar prothom e and independent variable guli following
#print(cleaned_DatasetOfCar_withDummies.columns.values)
reArrangedCol = ['Log_Price','Mileage','EngineV','Brand_BMW','Brand_Mercedes-Benz'
,'Brand_Mitsubishi','Brand_Renault','Brand_Toyota','Brand_Volkswagen'
,'Body_hatch','Body_other','Body_sedan','Body_vagon','Body_van'
,'Engine Type_Gas','Engine Type_Other','Engine Type_Petrol'
,'Registration_yes']

#HERE IS THE FINAL PREPOCESSED DATASET
print()
print("THE FINAL PREPOCESSED DATASET")
Preprocessed_DatasetOfCar = cleaned_DatasetOfCar_withDummies[reArrangedCol]
print(Preprocessed_DatasetOfCar.head())

##5. Standardizing(scaling features) the Dataset

#saperating the dependent and independent variables : targets = independent = y   |   features = dependent = x 
targets = Preprocessed_DatasetOfCar['Log_Price'] #only 'Log_Price' ta e target
features = Preprocessed_DatasetOfCar.drop(['Log_Price'],axis = 1) # 'Log_Price' k drop korar por je table seta purata e features

#importing necessary packages for feature scaling 
from sklearn.preprocessing import StandardScaler

#Creating a object of StandardScaler class
scaler = StandardScaler()

#telling to the object which variables are to be scaled
scaler.fit(features)

#features guli scale korar por arekta variable e rakhlam 
#Now all features are standardized 
scaled_features = scaler.transform(features)

##6. Now Spliting the Dataset into Training Testing Sets
x_train,x_test,y_train,y_test = train_test_split(scaled_features, targets, test_size = 0.2, shuffle = True, random_state = 42)

##7. Creating Regression
Regre = LinearRegression()

Regre.fit(x_train,y_train)

weights = Regre.coef_
bias = Regre.intercept_

print(bias)
print(weights)

y_hat = Regre.predict(x_train)
residuals = y_train - y_hat
#fig1 = sns.displot(residuals,kde=True)

#Checking the Goodness of the model 
RSquared = Regre.score(x_train,y_train)
print("0 <= RSquared <= 1")
print("R Squared =",RSquared.round(3))
print("75 % of the data fits to the model")

#Showing the summary of the Regression in a table
#RegreSummary = pd.DataFrame(data = Preprocessed_DatasetOfCar.columns.values, columns = ['Features'])
RegreSummary = pd.DataFrame()
RegreSummary['Features'] = features.columns.values
RegreSummary['Weights'] = weights
print(RegreSummary)
print("\"Brand Audy\" is missing so it is the Benchmark")
print("We have to explain the categorical data using this brand AUDY")

#showing the plot of y and yhat; 45 dergree angle proves that y and yhat are similar
#plt.scatter(y_train,y_hat)
#plt.xlabel('Targets(y_train)',size = 10)
#plt.ylabel('Predictions(y_hat)',size = 10)
#plt.show()

##8. Testing (The performance of the model)

#tesing data decchi amader model tar vitore
y_hat_test = Regre.predict(x_test)

#showing the plot of y_test and y_hat_test; 45 dergree angle proves that y and yhat are similar
plt.scatter(y_test,y_hat_test,alpha = 0.3)
plt.xlabel('Targets(y_test)',size = 10)
plt.ylabel('Predictions(y_hat_test)',size = 10)
plt.show()
#scatter plot theke amra dekhte pari je 45 degree line ei maximum data point obosthito

##akhon akta model performance er dataframe create korbo
#NOTE : amader je predicted value guli log transform kora so actual value pete setar exponential nite hobe
y_test = y_test.reset_index(drop = True)
model_performance = pd.DataFrame()
model_performance ['Price'] = np.exp(y_test)
model_performance ['Predicted Price'] = np.exp(y_hat_test)

model_performance ['Residuals'] = model_performance['Price'] - model_performance['Predicted Price']
model_performance ['Difference(%)'] = np.absolute(model_performance['Residuals']/(model_performance['Price']*100))

model_performance.sort_values(by = ['Difference(%)'])
print(model_performance)







