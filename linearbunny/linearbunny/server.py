"""This file connects main module with front end."""
from flask import Flask, render_template, request, redirect
import pandas as pd

from linearbunny import RetrieveExisting, SuccessForecast, Similarity, RetrieveCustomized, Strategy
import config as cg

# Create the application object
app = Flask(__name__)

#app = Flask(__name__, instance_path='/home/ubuntu/application')
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods=["GET", "POST"]) #we are now using these methods to get user input
def home_page():
    print('1 app root path is:', app.root_path)
    return render_template('index.html')  # render a template

@app.route('/output')
def recommendation_output():
    """Provide forecast from existing companies."""
    df_profile = pd.read_json(cg.profile)
    # Pull input
    some_input = request.args.get('user_input')

    # Case if empty
    if some_input == "":
        return render_template("index.html",
                               my_input=some_input,
                               my_form_result="Empty")
    else:
        some_output = " "
        number = int(some_input)
        my_sample = RetrieveExisting(number)
        x_test, y_test, company_index = my_sample.get_feature()
        my_forecast = SuccessForecast(x_test, y_test, number)
        prediction, actual, y_pred1 = my_forecast.deep_learning_existing()
        similar_companies = Similarity(my_company_index=company_index,
                                       customized_feature=False)
        image_name_top, url_top = similar_companies.similarity()
        prediction, actual, y_pred1 = my_forecast.deep_learning_existing()
        some_prediction = 'Company ' + some_input + ' Click here for prediction of more companies.'
        target_image = 'pic_' + str(company_index+1)

        my_image1 = image_name_top[0]
        my_image2 = image_name_top[1]
        my_image3 = image_name_top[2]
        my_image4 = image_name_top[3]
        my_image5 = image_name_top[4]
        my_image6 = image_name_top[5]
        my_url1 = url_top[0]
        my_url2 = url_top[1]
        my_url3 = url_top[2]
        my_url4 = url_top[3]
        my_url5 = url_top[4]
        my_url6 = url_top[5]

        profile = df_profile.iloc[:, 0][company_index]
        print(my_image1, my_image2, my_image3, my_image4, my_image5, my_image6)
        if prediction == 'Success' and actual == 'Success':
            classification_pic = 'classification_psas'
        elif prediction == 'Success' and actual == 'Failure':
            classification_pic = 'classification_psaf'
        elif prediction == 'Failure' and actual == 'Failure':
            classification_pic = 'classification_pfaf'
        elif prediction == 'Failure' and actual == 'Success':
            classification_pic = 'classification_pfas'
        new_strategy = Strategy(my_company_index=company_index,
                                my_prediction=prediction, my_actual=actual, my_number=number)
        strategy_outcome = new_strategy.strategy()
        if strategy_outcome == 0:
            strategy_interactive = ''
            strategy_pic = 'need_customized_strategy1.png'
            strategy_text = ''

        elif strategy_outcome == 1:
            strategy_interactive = 'strategy_forecast'
            strategy_pic = 'strategy_sorted.png'
            strategy_text = 'Click here for interactive plot showing strategy forecast.'
        else:
            strategy_interactive = ''
            strategy_pic = 'congrats.gif'
            strategy_text = 'Congratulations! The target company is successful!'

        return render_template("index.html",
                               my_input=some_input,
                               my_output=some_output,
                               my_classification_pic=classification_pic,
                               my_target_image=target_image,
                               my_image11=my_image1,
                               my_image22=my_image2,
                               my_image33=my_image3,
                               my_image44=my_image4,
                               my_image55=my_image5,
                               my_image66=my_image6,
                               my_url11=my_url1,
                               my_url22=my_url2,
                               my_url33=my_url3,
                               my_url44=my_url4,
                               my_url55=my_url5,
                               my_url66=my_url6,
                               my_profile=profile,
                               my_strategy_pic=strategy_pic,
                               my_strategy_interactive=strategy_interactive,
                               my_strategy_text=strategy_text,
                               my_form_result="NotEmpty",
                               my_prediction=some_prediction)

@app.route('/customize', methods=['POST'])
def customize():
    """Provide customized forecast."""
    input_dic = {}
    company_name = request.form['company_name']
    investment_rounds = request.form['investment_rounds']
    funding_round = request.form['funding_rounds']
    year_founded = request.form['year_founded']
    last_funding_year = request.form['last_funding_year']
    funding_total_usd = request.form['funding_total_usd']
    milestone = request.form['milestone']
    relationship = request.form['relationship']
    profile = request.form['company_profile']
    funding_types = request.form.getlist('funding_types')
    input_dic = {'company_name':company_name,
                 'investment_rounds':investment_rounds,
                 'funding_rounds':funding_round,
                 'year_founded':year_founded,
                 'last_funding_year':last_funding_year,
                 'funding_total_usd':funding_total_usd,
                 'funding_types':funding_types,
                 'milestones':milestone,
                 'relationships':relationship,
                 'domain':profile}
    input_df = pd.DataFrame(input_dic)
    print(input_df)
    input_df.to_csv('exploratory_data/input_df.csv')
    print(input_dic)

    if input_dic != {}:
        my_sample = RetrieveCustomized()
        new_feature = my_sample.get_feature()
        x_test_input, input_np = my_sample.feature_nlp(new_feature)
        my_forecast = SuccessForecast(x_test_input, y_test=[0], number=0)
        print(my_forecast.deep_learning_customized())
        prediction1 = my_forecast.deep_learning_customized()
        new_strategy = Strategy(my_company_index=False,
                                my_prediction=prediction1, my_actual=False, my_number=False)
        strategy_outcome = new_strategy.strategy_customize(my_prediction1=prediction1,
                                                           my_input=input_np)
        similar_companies = Similarity(my_company_index=False, customized_feature=input_np)
        image_name_top, url_top = similar_companies.similarity()

        my_image1 = image_name_top[0]
        my_image2 = image_name_top[1]
        my_image3 = image_name_top[2]
        my_image4 = image_name_top[3]
        my_image5 = image_name_top[4]
        my_image6 = image_name_top[5]
        my_url1 = url_top[0]
        my_url2 = url_top[1]
        my_url3 = url_top[2]
        my_url4 = url_top[3]
        my_url5 = url_top[4]
        my_url6 = url_top[5]

        if prediction1 == 'Success':
            classification_pic = 'classification_ps'
            strategy_interactive = ''
            strategy_pic = 'congrats.gif'
            strategy_text = 'Congratulations! The target company is successful!'
        if prediction1 == 'Failure':
            classification_pic = 'classification_pf'
            strategy_text = 'strategy_forecast'
            strategy_pic = 'strategy_sorted.png'

        if strategy_outcome == 0:
            strategy_interactive = ''
            strategy_pic = 'need_customized_strategy1.png'
            strategy_text = ''

        elif strategy_outcome == 1:
            strategy_interactive = 'strategy_forecast'
            strategy_pic = 'strategy_sorted.png'
            strategy_text = 'Click here for interactive plot showing strategy forecast.'
        else:
            strategy_interactive = ''
            strategy_pic = 'congrats.gif'
            strategy_text = 'Congratulations! The target company is successful!'

        return render_template("index.html",
                               test_test=prediction1,
                               my_classification_pic=classification_pic,
                               my_strategy_pic=strategy_pic,
                               my_strategy_text=strategy_text,
                               my_strategy_interactive=strategy_interactive,
                               my_image11=my_image1,
                               my_image22=my_image2,
                               my_image33=my_image3,
                               my_image44=my_image4,
                               my_image55=my_image5,
                               my_image66=my_image6,
                               my_url11=my_url1,
                               my_url22=my_url2,
                               my_url33=my_url3,
                               my_url44=my_url4,
                               my_url55=my_url5,
                               my_url66=my_url6)

    return redirect('/')

# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(threaded=False, debug=True) #will run locally http://127.0.0.1:5000/
    #app.run(host='0.0.0.0', debug = True) # running on aws
