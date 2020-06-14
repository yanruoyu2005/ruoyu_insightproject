
from flask import Flask, render_template, request
from just_test_v2 import get_profile,test_single, strategy
import pandas as pd

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

# Create the application object
app = Flask(__name__)

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output')
def recommendation_output():       
       df_profile = get_profile()
       # Pull input
       some_input =request.args.get('user_input')            
       
       # Case if empty
       if some_input =="":
           return render_template("index.html",
                                  my_input = some_input,
                                  my_form_result="Empty")
       else:
           some_output="yeay!"
           number = int(some_input)
           prediction, actual, company_index,image_name_top,url_top = test_single(number)
           some_prediction = 'Company ' +  some_input + ': ' + 'Prediction: ' + prediction + '__' + '  Actual: ' + actual
           #print ('Predicted outcome, ', prediction)
           #print ('Actual outcome, ', actual)
           #some_image="giphy.gif"
           #some_image="confusion_matrix.png"
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

           profile = df_profile.iloc[:,0][company_index] 

           print(my_image1,my_image2,my_image3,my_image4,my_image5,my_image6)

           if prediction == 'Success' and actual == 'Success':
               classification_pic = 'classification_psas'
           elif prediction == 'Success' and actual == 'Failure':
               classification_pic = 'classification_psaf'
           elif prediction == 'Failure' and actual == 'Failure':
               classification_pic = 'classification_pfaf'
           elif prediction == 'Failure' and actual == 'Success':
               classification_pic = 'classification_pfas'

           
           strategy_outcome = strategy(company_index,prediction,actual,number)
           if strategy_outcome==0:
               strategy_interactive = ''
               strategy_pic = 'need_customized_strategy1.png'
               strategy_text = ''

           elif strategy_outcome==1:
               strategy_interactive = 'strategy_forecast'
               strategy_pic = 'strategy_sorted.png'
               strategy_text = 'Click here for interactive plot showing strategy forecast.'
           else:
               strategy_interactive=''
               strategy_pic = 'congrats.gif'
               strategy_text = 'Congratulations! The target company is successful!'

           return render_template("index.html",
                              my_input=some_input,
                              my_output=some_output,
                              my_classification_pic=classification_pic,
                              #my_number=some_number,
                              #my_img_name=some_image,
                              #my_domain = 'gehealthcare.com?size=200&',
                              my_target_image = target_image,
                              my_image11 = my_image1,
                              my_image22 = my_image2,
                              my_image33 = my_image3,
                              my_image44 = my_image4,
                              my_image55 = my_image5,
                              my_image66 = my_image6,
                              
                              my_url11 = my_url1,
                              my_url22 = my_url2,
                              my_url33 = my_url3,
                              my_url44 = my_url4,
                              my_url55 = my_url5,
                              my_url66 = my_url6,

                              my_profile = profile,

                              my_strategy_pic = strategy_pic,
                              my_strategy_interactive = strategy_interactive,
                              my_strategy_text = strategy_text,
                             
                              my_form_result="NotEmpty",
                              my_prediction = some_prediction)

# start the server with the 'run()' method
if __name__ == "__main__":


    #app.run(threaded=False, debug=True) #will run locally http://127.0.0.1:5000/
    app.run(host='0.0.0.0', debug = True) # try running on aws


