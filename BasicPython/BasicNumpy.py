import time
import numpy as np
import pandas as pd
import urllib.request as urlreq

#def cpuBegin():
#    # get the start time
#    st = time.process_time()
#    return st

#def cpuEnd():
#    # get the start time
#    et = time.process_time()
#    return et

#arr1 = [0.1,0.2,0.3]
#arr2 = [10,20,30]

###dot multiplication of 1d array: arr1 and arr2
##process1:
#result = 0
#for i in range(len(arr1)):
#    result = result + (arr1[i]*arr2[i])
#print(result)

##process2:
#result = 0
#for x1,x2 in zip(arr1,arr2):
#    result = result + (x1*x2)
#print(result)

##process3[using np]:
#result = np.dot(arr1,arr2)
#print(result)

##mutrix multiplication of n * 1 dimension
climate_data = [[73,67,43],
                [91,88,64],
                [87,134,58],
                [102,43,37],
                [69,96,70]]
weight = [0.3,0.2,0.5]
result = np.matmul(climate_data,weight)
#print(result)

pd.set_option('display.max_columns', None) 
climate_dataset = pd.read_csv('./DailyDelhiClimateTrain.csv')
#print(climate_dataset)
#climate_dataset = pd.read_csv('./DailyDelhiClimateTrain.csv',header=False,index = False)
#climate_dataset.drop(columns='',axis=1,inplace=True)
#climate_dataset = climate_dataset.to_csv(header=None, index= False)
climate_dataset.iloc[1: , 2:]
print(climate_dataset.iloc[1: , 3:])



