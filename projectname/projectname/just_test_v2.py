import os
import sys

from keras.layers import Dense
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import loadtxt
from numpy import savetxt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import itertools



def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    pic_name = path1 + '/static/assets/img/portfolio/confusion_matrix1.png'
    plt.savefig(pic_name)
    plt.close()
    
    
def test_single(number):
    global path1
    path1 = get_script_path()
    path_data = path1 + '\exploratory_data'
    classifier=load_model(path_data + '\pretrain_model_add_domain_binary_keywords_id.hdf5')
    X_test_pd = pd.read_csv(path_data + '\X_test_pd_v3_id.csv')

    company_index = X_test_pd.iloc[:,0][number]
    print('company_index is',company_index)

    X_test_pd_feature = X_test_pd.iloc[:,1:].iloc[:,:-1]
    X_test = np.asarray(X_test_pd_feature)
    
    y_test_pd = pd.read_csv(path_data + '\y_test_pd_v3.csv')
    y_test = np.asarray(y_test_pd.iloc[:,1:])

    y_pred=classifier.predict(X_test)
    y_pred1 =(y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred1)
    if y_pred[number] > 0.5:
        prediction = 'Success'
    if y_pred[number] < 0.5:
        prediction = 'Failure'
    if y_test[number] == 1:
        actual = 'Success'
    if y_test[number]==0:
        actual = 'Failure'   
    print ('Predicted outcome, ', prediction)
    print ('Actual outcome, ', actual)
    cm_plot_labels = ['Faliure','Success']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    return prediction, actual, company_index

# Generate an array represent all possible combinations of factors for each factor
def generate_array():
    # x, y, z represents funding_total_usd, relationship and timeline
    # Generate a gradient of factors
    factor_list = range(5,41,2)
    factor = np.asarray(factor_list)/10
    array = []
    for x in factor:
        for y in factor:
            for z in factor:
                array.append([x, y, z])
    array = np.asarray(array)
    array_df = pd.DataFrame(array)
    deviation=[]
    for i in range(array.shape[0]):
        deviation.append(abs(array[i,0]-1) + abs(array[i,1]-1) + abs(array[i,2]-1))
    array_df['deviation'] = deviation
    array_df.to_csv(path1 + '\\exploratory_data\\factor_deviation.csv')
    return array

# Based on user input company ID, obtain its original feature 
def obtain_target(company_index,array):
    df = pd.read_csv('C:\\Users\\yanru\\Desktop\\crunbase2013\\feature_one_hot1_0_1.csv')
    x_target = list(df.iloc[company_index,3:])
    size = array.shape[0]

    target_list=[]
    for i in range(size):
        target_list.append(x_target)
    target_array = np.asarray(target_list)
    return target_array, x_target

# Generate a new array with hypothetical scaled feature based on the gradient of factors
def scale_array(company_index):
    array = generate_array()
    target_array, x_target = obtain_target(company_index,array)
    print('target',x_target)
    x_target_temp = pd.DataFrame(x_target)       
    x_target_temp.to_csv(path1 + '\\exploratory_data\\x_target_temp.csv')
    array[:,0] = array[:,0]*(x_target[2]+10000)
    array[:,1] = array[:,1]*(x_target[4]+1)
    array[:,2] = array[:,2]*(x_target[25]+0.1)
    new_array = array
    target_array[:,[2,4,25]] = new_array
    #print('new strategy feature prior to standard_scaler',target_array)
    x_target_temp.to_csv(path1 + '\\exploratory_data\\x_target_temp.csv')
    file_name = path1 + '\\exploratory_data\\strategy_feature_temp.csv'
    savetxt(file_name, target_array, delimiter=',')
    return new_array,target_array

def test_strategy(target_array,company_index):
    path_data = path1 + '\exploratory_data'
    classifier=load_model(path_data + '\pretrain_model_add_domain_binary_keywords_id.hdf5')

    data = loadtxt(path_data+ '\whole_feature.csv', delimiter=',')
    X_test_number_nlp = list(data[company_index,:][57:])
    nlp_array = []
    for i in range(target_array.shape[0]):
        nlp_array.append(X_test_number_nlp)
    nlp_array=np.asarray(nlp_array)
    X_test = np.concatenate((target_array,nlp_array),axis=1)
  
    scaler_file = path_data + '\scaler.pkl'
    sc = pickle.load(open(scaler_file,'rb'))
    X_test = sc.fit_transform(X_test)
    
    y_test = np.asarray([1]*target_array.shape[0])
    y_pred=classifier.predict(X_test)
    y_pred1 =(y_pred>0.5)
    cm_strategy = confusion_matrix(y_test, y_pred1)
    print('confusion_matrix_new_strategy',cm_strategy)
    return cm_strategy,y_pred1

def strategy(company_index,prediction,actual,number):
    if actual=='Success':
        print('congrats')
    if actual=='Failure':
        new_array,target_array=scale_array(company_index)
        cm_strategy,y_pred1 = test_strategy(target_array,company_index)
        if cm_strategy[1,1]==0:
            print('The generic strategy does not suit you, please contact us for developing customized strategy')
        else:
            print('We can help you succeed!')
            factor_deviation_df = pd.read_csv(path1 + '\\exploratory_data\\factor_deviation.csv')
            factor_deviation_df['strategy_prediction'] = y_pred1
            factor_deviation_df.to_csv(path1 + '\\exploratory_data\\factor_deviation_pre.csv')
            factor_true = factor_deviation_df[factor_deviation_df['strategy_prediction']==True]
            factor_true_sort = factor_true.sort_values(by=['deviation'])
            factor_true_sort.to_csv(path1 + '\\exploratory_data\\factor_true_sort.csv')
            data = factor_true_sort.iloc[0:10,1:4]
            data.columns = ['Total Funding', 'Relationship', 'Timeline']
            
            strategy_list=[]
            for i in range(10):
                strategy = 'Strategy ' + str(i+1)
                strategy_list.append(strategy)
            data.index = strategy_list

            plt.clf()
            fig = plt.figure(figsize=(8,10))
            title_name = 'Scaling Factors of Company ' + str(number)
            plt.rcParams['font.family'] = "serif"
            sns.set_context(font_scale=1.6)
            sns_plot=sns.heatmap(data, cmap='coolwarm', annot=True, fmt=".1f",annot_kws={'size':16})
            sns_plot.tick_params(labelsize=14)
            sns_plot.axes.set_title(title_name,fontsize=20)
            plt.yticks(rotation=0) 
            plt.tight_layout()
            pic_name = path1 + '/static/assets/img/portfolio/strategy_sorted.png'
            plt.savefig(pic_name)
            plt.close()