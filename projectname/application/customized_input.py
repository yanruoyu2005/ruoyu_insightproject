import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from numpy import savetxt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

from bunny_functions import generate_array
import strategy_interactive_plot 

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def get_input_feature():
    global path
    #path = get_script_path() # previously used for local server
    path = '/home/ubuntu/application' # used for production server
    new_list_edited=pd.read_json (path + '/exploratory_data/bt_c2to4_edited1.txt')
    new_list_pd = new_list_edited
    new_list_pd.columns = ['domain']
    text = list(new_list_edited.iloc[:,0]) # historical profile text
    key_words = pd.read_csv(path + '/exploratory_data/key_words.csv')
    keywords_list = key_words.iloc[0:,].values.reshape(1,-1).tolist()[0] # domain key words
    feature_pd = pd.read_csv(path + '/exploratory_data/feature_one_hot1_0_1.csv')
    feature_text_pd = feature_pd
    feature_text_pd['domain'] = new_list_pd['domain']
    feature_text_pd = feature_text_pd.iloc[:,3:] # historical feature with profile
    df = pd.read_csv(path + '/exploratory_data/input_df.csv') # customized input
    column_names = feature_text_pd.columns
    df1 = pd.DataFrame(np.zeros([1, 58]), columns = column_names) # framework for feature of customized input
    # Assign input values to framework
    df1.investment_rounds = df.investment_rounds[0]
    df1.funding_rounds = df.funding_rounds[0]
    df1.funding_total_usd = df.funding_total_usd[0]
    df1.milestones = df.milestones[0]
    df1.relationships = df.relationships[0] 
    df1.timeline = df.last_funding_year[0] - df.year_founded[0]
    df1.a = df[df.funding_types=='Series a'].shape[0]
    df1.b = df[df.funding_types=='Series b'].shape[0]
    df1.c = df[df.funding_types=='Series c'].shape[0]
    df1.d = df[df.funding_types=='Series d'].shape[0]
    df1.e = df[df.funding_types=='Series e'].shape[0]
    df1.f = df[df.funding_types=='Series f'].shape[0]
    df1.g = df[df.funding_types=='Series g'].shape[0]
    df1.angel = df[df.funding_types=='Angel'].shape[0]
    df1.convertible = df[df.funding_types=='Convertible'].shape[0]
    df1.crowd = df[df.funding_types=='Crowd'].shape[0]
    df1.crowd_equity = df[df.funding_types=='Crowd Equity'].shape[0]
    df1.debt_round = df[df.funding_types=='debt_round'].shape[0]
    df1.grant = df[df.funding_types=='Grant'].shape[0]
    df1.partial = df[df.funding_types=='Partial'].shape[0]
    df1.post_ipo_debt = df[df.funding_types=='Post IPO Debt'].shape[0]
    df1.post_ipo_equity = df[df.funding_types=='Post IPO Equity'].shape[0]
    df1.private_equity = df[df.funding_types=='Private Equity'].shape[0]
    df1.secondary_market = df[df.funding_types=='Secondary Market'].shape[0]
    df1.seed = df[df.funding_types=='Seed'].shape[0]
    df1.unattributed = df[df.funding_types=='Unattributed'].shape[0]
    df1.domain = df.domain[0]
    # df1 now contains one row of input feature
    new_feature = feature_text_pd.append(df1)
    # Use customized NLP to prepare features
    column_trans = ColumnTransformer([('domain_bow', 
                                   CountVectorizer(text, 
                                   analyzer='word', 
                                   binary=True, 
                                   decode_error='strict',
                                   lowercase=True, 
                                   max_features=None, 
                                   max_df=0.80,
                                   ngram_range=(1, 2), 
                                   preprocessor=None, 
                                   stop_words=['english','company','based','founded','and','the','in','of','for','to','is','as','inc','with','that','was','its','it','has','on','on','by','an','which','are'], 
                                   strip_accents=None, 
                                   tokenizer=None, 
                                   vocabulary=keywords_list), 
                                   'domain')], 
                                   remainder='passthrough')

    column_trans.fit(feature_text_pd)
    new_array1=column_trans.transform(new_feature).toarray()

    sc = pickle.load(open(path + '/exploratory_data/scaler.pkl','rb'))
    input_np=new_array1[570].reshape(1,-1)
    X_test = sc.transform(input_np) # X_test_input is the final feature that is processed by NLP, feature engineering and normalization
    X_test_input = X_test
    global company_name1
    company_name1 = df.company_name[0]
    return X_test_input,df1,input_np

def test_input():
    X_test,df1,input_np = get_input_feature()
    image_name_top,url_top = similarity_input(input_np)
    classifier=load_model(path + '/exploratory_data/pretrain_model_add_domain_binary_keywords_id.hdf5')
    y_pred=classifier.predict(X_test)
    y_pred1 =(y_pred>0.5)
    if y_pred[0][0] > 0.5:
        prediction = 'Success'
    else:
        prediction = 'Failure'
        raw_input = list(df1.iloc[0,:])
        print('raw_input',raw_input)
        strategy_input(raw_input,input_np,y_pred1)    
    print ('Predicted outcome, ', prediction)
    return prediction,image_name_top,url_top

def strategy_input(raw_input,input_np,y_pred1):
    array = generate_array()
    size = array.shape[0]
    target_list=[]
    input1 = raw_input[0:57] + input_np.tolist()[0][57:]
    for i in range(size):
        target_list.append(input1)
    target_array = np.asarray(target_list)
    print(target_array.shape)

    array[:,0] = array[:,0]*(raw_input[2]+1)
    array[:,1] = array[:,1]*(raw_input[4]+0.05)
    array[:,2] = array[:,2]*(raw_input[25]+0.05)

    new_array = array
    target_array[:,[2,4,25]] = new_array

    file_name = path + '/exploratory_data/strategy_feature_temp.csv'
    savetxt(file_name, target_array, delimiter=',')

    sc = pickle.load(open(path + '/exploratory_data/scaler.pkl','rb'))
    X_test = sc.transform(target_array)

    classifier=load_model(path + '/exploratory_data/pretrain_model_add_domain_binary_keywords_id.hdf5')
    y_pred=classifier.predict(X_test)
    y_pred1 =(y_pred>0.5)
    y_test = np.asarray([1]*target_array.shape[0])

    cm_strategy = confusion_matrix(y_test, y_pred1)
    print('confusion_matrix_new_strategy',cm_strategy)
    

    if cm_strategy.shape == (1,1):
        print('all strategies successful')
        make_plots(y_pred1)

    if cm_strategy.shape == (2,2):
        if cm_strategy[1,1]==0:           
            print('The generic strategy does not suit you, please contact us for developing customized strategy')   
        else:
            print('some strategies are successful')

def make_plots(y_pred1):
    # sort strategy, generate table, 3D plots and similarity
    factor_deviation_df = pd.read_csv(path + '/exploratory_data/factor_deviation.csv')
    factor_deviation_df['strategy_prediction'] = y_pred1
    factor_deviation_df.to_csv(path + '/exploratory_data/factor_deviation_pre.csv')
    factor_true = factor_deviation_df[factor_deviation_df['strategy_prediction']==True]
    factor_true_sort = factor_true.sort_values(by=['deviation'])
    factor_true_sort.to_csv(path + '/exploratory_data/factor_true_sort.csv')

    data = factor_true_sort.iloc[0:10,1:4]
    data.columns = ['Total Funding', 'Relationship', 'Timeline']
                
    strategy_list=[]
    for i in range(10):
        strategy = 'Strategy ' + str(i+1)
        strategy_list.append(strategy)
    data.index = strategy_list   
    # Make heat map
    plt.clf()
    fig = plt.figure(figsize=(8,10))
    title_name = 'Scaling Factors of Company ' + company_name1
    plt.rcParams['font.family'] = "serif"
    sns.set_context(font_scale=1.6)
    sns_plot=sns.heatmap(data, cmap='coolwarm', annot=True, fmt=".1f",annot_kws={'size':16})
    sns_plot.tick_params(labelsize=14)
    sns_plot.axes.set_title(title_name,fontsize=20)
    plt.yticks(rotation=0) 
    plt.tight_layout()
    pic_name = path + '/static/assets/img/portfolio/strategy_sorted.png'
    plt.savefig(pic_name)
    plt.close()

def similarity_input(input_np):
    whole_feature = loadtxt(path+ '/exploratory_data/whole_feature.csv', delimiter=',')
    similarity_list=[]
    for single_feature in whole_feature:
        single_feature=np.reshape(single_feature,(1,-1))
        cosine_np = cosine_similarity(input_np,single_feature)
        similarity_list.append(cosine_np[0][0])
    cosine_df = pd.DataFrame(similarity_list)
    cosine_df.columns=['cosine_score']
    cosine_df['numeric_index']=list(range(whole_feature.shape[0]))
    cosine_df_sorted=cosine_df.sort_values(by=['cosine_score'],ascending=False)
    print(cosine_df_sorted.head(20))
    similarity_top = list(cosine_df_sorted['numeric_index'][0:6])
    print(similarity_top)
    df_logo = pd.read_csv(path + '/exploratory_data/company_logo.csv')
    df_url=df_logo.homepage_url
    url_top = []
    for numeric_index in similarity_top:
        my_url = df_url[numeric_index]
        if str(my_url) != 'nan':
            my_url = my_url[1:][:-1]
            url_top.append(my_url)
        else:
            url_top.append('nan')
    print(url_top)
    image_name_top = []
    for numeric_index in similarity_top:
        image_name = 'pic_'+ str(numeric_index+1) + '.png'
        image_name_top.append(image_name)

    return image_name_top,url_top






    
            





















    






    


    

    
    
















