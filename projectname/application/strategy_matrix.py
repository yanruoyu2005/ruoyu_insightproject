import numpy as np
import pandas as pd



# Generate an array represent all possible combinations of factors for each factor
def generate_array():
    # x, y, z represents funding_total_usd, relationship and timeline
    # Generate a gradient of factors
    factor_list = range(5,21,1)
    factor = np.asarray(factor_list)/10
    array = []
    for x in factor:
        for y in factor:
            for z in factor:
                array.append([x, y, z])
    array = np.asarray(array)
    return array

# Based on user input company ID, obtain its original feature 
def obtain_target(number,array):
    df = pd.read_csv('C:\\Users\\yanru\\Desktop\\crunbase2013\\feature_one_hot1_0_1.csv')
    x_target = list(df.iloc[number,3:])
    size = array.shape[0]

    target_list=[]
    for i in range(size):
        target_list.append(x_target)
    target_array = np.asarray(target_list)
    return target_array, x_target

# Generate a new array with hypothetical scaled feature based on the gradient of factors
def scale_array(number):
    array = generate_array()
    target_array, x_target = obtain_target(number,array)
    print(x_target)
    array[:,0] = array[:,0]*x_target[2]
    array[:,1] = array[:,1]*x_target[4]
    array[:,2] = array[:,2]*x_target[25]
    print(array)
    new_array = array
    return new_array,target_array

new_array,target_array = scale_array(2)
print(new_array)
target_array[:,[2,4,25]] = new_array
print(target_array[:,[2,4,25]])
