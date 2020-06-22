"""Make interactive plot"""
import chart_studio.plotly as py
import pandas as pd
import plotly
import plotly.graph_objs as go

import config as cg


def get_plot():
    """Make the plot"""
    path = cg.path
    path_data = path + '/exploratory_data'
    #path1 = '/home/ubuntu/application'
    df = pd.read_csv(path_data + '/strategy_feature_temp.csv', header=None)
    df_y = pd.read_csv(path_data + '/factor_deviation_pre.csv')
    outcome = []
    for i in range(df.shape[0]):
        if str(df_y.strategy_prediction[i]) == 'True':
            outcome.append('Success')
        else:
            outcome.append('Failure')

    df_s = df.iloc[:, [2, 4, 25]]
    df_s['forecast'] = outcome
    df_s['deviation'] = list(df_y.deviation)
    print(df_s)

    modified_df = df_s
    modified_df.index = list(df_s['forecast'])

    forecast_class = ['Success', 'Failure']
    color_class = ['green', 'red']

    class_number = len(forecast_class)
    dataframes = []

    for sample in forecast_class:
        dataframes.append(modified_df.loc[sample])

    data = []
    for i in range(class_number):
        dot_size = 7 + (dataframes[i]['deviation'])*0.1
        trace1 = go.Scatter3d(x=dataframes[i].iloc[:, 0],
                              y=dataframes[i].iloc[:, 1],
                              z=dataframes[i].iloc[:, 2],
                              name=forecast_class[i],
                              line=dict(width=0),
                              mode='markers',
                              marker=dict(size=dot_size,
                                          color=color_class[i]),)
    data.append(trace1)
    layout = go.Layout(title="Strategy Forecast",
                       autosize=True,
                       scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode="manual",
                                  xaxis=dict(title='x - Funding Total USD'),
                                  yaxis=dict(title='y - Relationship'),
                                  zaxis=dict(title='z - Timeline')))
    fig = dict(data=data, layout=layout)
    save_name = path + '/static/assets/img/portfolio/strategy_forecast.html'
    plotly.offline.plot(fig, filename=save_name)
