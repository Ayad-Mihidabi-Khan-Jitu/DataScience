import numpy  as nmpy
import matplotlib.pyplot as plt
import pandas as pd
print("Here")

dataset= pd.read_csv('C:/Users/HP 840 G1/Documents/Visual Studio 2022/Projects/DataScience/DataScienceInPython/Drawable/DataSet.csv')
X=dataset.iloc[:,1:2].values
print(X)

dataset.plot()
plt.show()

s=pd.Series([1,2,3,"f"])
date=pd.date_range('20200829',periods=6)
print(date)

df=pd.DataFrame(nmpy.random.randn(6,4),index=date,columns=list('ABCD'))
print(df)

df['E']=['One','Two','Three','Four','Five','Six']
print(df)

s1=pd.Series([1,2,3,4,5,6],index=pd.date_range('20200904', periods=6))
print(s1)

df['F']=s1
print(df)

df.to_excel('dateSet.xlsx',sheet_name='sheet1')
df1=pd.read_excel('dateSet.xlsx',sheet_name='sheet1',index_col=None,na_values=['Na'])
print(df1)

#df=df.cumsum()
df.plot()
plt.show()