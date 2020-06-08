import os
import sys

from keras.layers import Dense
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    
    
def test_single(number):
    global path1
    path1 = get_script_path()
    path_data = path1 + '\exploratory_data'
    classifier=load_model(path_data + '\pretrain_model_add_domain_binary_keywords_id.hdf5')
    X_test_pd = pd.read_csv(path_data + '\X_test_pd_v3_id.csv')
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
    return prediction, actual
