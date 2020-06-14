import os
import sys

import chart_studio.plotly as py
import numpy as np
from numpy import loadtxt
import pandas as pd
import plotly
import plotly.graph_objs as go


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_plot():
    global path1
    path1 = get_script_path()
    path_data = path1 + '/exploratory_data'
    df = pd.read_csv(path_data + '/strategy_feature_temp.csv',header=None)
    df_y = pd.read_csv(path_data + '/factor_deviation_pre.csv')
    
    outcome=[]
    for i in range(df.shape[0]):
        if str(df_y.strategy_prediction[i])=='True':
            outcome.append('Success')
        else:
            outcome.append('Failure')

    x = df.iloc[:,2]
    y = df.iloc[:,4]
    z = df.iloc[:,25]
    df_s = df.iloc[:,[2,4,25]]
    df_s['forecast'] = outcome
    df_s['deviation'] = list(df_y.deviation)
    print(df_s)

    modified_df = df_s
    modified_df.index = list (df_s['forecast'])

    forecast_class = ['Success', 'Failure']
    color_class = ['green', 'red']

    class_number = len (forecast_class)   
    dataframes = []
    
    for sample in forecast_class:
        dataframes.append (modified_df.loc[sample])


# Define x, y, z for trace for each subgroup, and append all trace to a list called "data"
    data = []
    for i in range (class_number):
        dot_size = 7 + (dataframes [i]['deviation'])*0.1                   
        trace1 = go.Scatter3d (x = dataframes[i].iloc[:,0], y = dataframes[i].iloc[:,1], z = dataframes[i].iloc[:,2],
                        name = forecast_class [i],
                       line = dict(width=0),
                       mode = 'markers',
                       marker = dict(size = dot_size, color = color_class [i]),                       )
        
        data.append (trace1)
    
    layout = go.Layout ( title = "Strategy Forecast",
                         autosize = True,
                         scene = dict (aspectratio = dict (x = 1, y = 1, z = 1), aspectmode = "manual",
                         xaxis=dict(title='x - Funding Total USD'),
                         yaxis=dict(title='y - Relationship'),
                         zaxis=dict(title='z - Timeline')
                         )
                         )


    
    fig = dict (data = data, layout = layout)
                         

# E. Save tsne as html file and specify file name.
    save_name =  path1 + '/static/assets/img/portfolio/strategy_forecast.html'
    plotly.offline.plot (fig, filename = save_name)
    
