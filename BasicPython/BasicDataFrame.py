###akhon akta jinish shikhbo seta holo: data frame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
#jaha data frame tahai table
#SYNTEX1 : dataFrame_Name = pd.DataFrame(data = [numerical or categorical], columns = ['Column_Name'])
#SYNTEX2 : dataFrame_Name = pd.DataFrame({'Column_Name1': [values], 'Column_Name2': [values]})

#creating dataFrame
populationData = pd.DataFrame(data = [2,4,7,9], columns = ['People'])

#creating dataFrame
populationData1 = pd.DataFrame({'Time': [7.12,8,9,10]})

#printing dataframe
print(populationData)

#printing dataframe
print(populationData1)

#joining the two dataframe into one data frame
MarketData = populationData.join(populationData1)

#printing the dataframe
print(MarketData)

#renaming a column of a dataframe
MarketData = MarketData.rename(index={0: 'Saturday', 1: 'Sunday', 2: 'Monday', 3: ' Tuesday'}) 

#maping the values of a column
MarketData['Time'] = MarketData['Time'].map({7: 'Very Early', 8: 'Early', 9: 'Late' ,10: 'Very Late'}) 

#printing the dataframe
print(MarketData)

#converting the dataframe into excel and save it to local storage
#SYNTEX : dataframe_name.to_excel('file_name.xlsx',sheet_name='sheet_name')
MarketData.to_excel('MarketData.xlsx',sheet_name='sheet1')

##Akhon dekhbo kivabe dataframe k customize kora jay
#show only all the columns
pd.set_option('max_columns', None)
print(MarketData)

#column resetting as user defined '1'
pd.set_option('max_columns', 1)
print(MarketData)

#reset the default 
pd.reset_option('max_columns')
print(MarketData)

#column-width resetting
pd.set_option('max_colwidth', 10)
print(MarketData)

#precision resetting of the dataframe user defined '2'
pd.set_option('precision', 1)
print(MarketData)

#show only all the rows
pd.set_option('max_rows', None)
print(MarketData)

#row resetting as user defined '2'
pd.set_option('max_rows', 2)
print(MarketData)

#reset the default 
pd.reset_option('max_rows')
print(MarketData)

#prints first five rows of the dataset
print("first five rows of the dataset")
print(MarketData.head())

#prints first 2 rows of the dataset
print("first two rows of the dataset")
print(MarketData.head(2))
