from flask import Flask, render_template, request
from just_test_v2 import test_single 



# Create the application object
app = Flask(__name__)

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output')
def recommendation_output():
#       

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
           prediction,actual = test_single(number)
           some_prediction = 'Prediction: ' + prediction + '__' + '  Actual: ' + actual
           #print ('Predicted outcome, ', prediction)
           #print ('Actual outcome, ', actual)
           #some_image="giphy.gif"
           #some_image="confusion_matrix.png"
           return render_template("index.html",
                              my_input=some_input,
                              my_output=some_output,
                              #my_number=some_number,
                              #my_img_name=some_image,
                              my_form_result="NotEmpty",
                              my_prediction = some_prediction)


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(threaded=False, debug=True) #will run locally http://127.0.0.1:5000/


