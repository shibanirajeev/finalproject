from flask import Flask, request, render_template, redirect, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from textblob import TextBlob
import sklearn.externals
import joblib

app = Flask(__name__, static_url_path="") 

scaler = joblib.load( "scaler.save")
modelIE = load_model("modelIE.h5")
modelNS = load_model("model_testNS.h5")

@app.route('/')
def home():
    return render_template('background.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route("/image/<path:path>")
def image(path):
    return send_from_directory('templates/', path)

#??? why will go to 
@app.route('/predict',methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('prediction.html')
    if request.method == 'POST':

        input_features = [str(x) for x in request.form.to_dict().values()]
        combined_features = [input_features[0]+ ' '+input_features[1]+ ' '+ input_features[2]]
        print(combined_features)
        df_test = pd.DataFrame({'posts':combined_features})
        df_test['words_per_comment'] = df_test['posts'].apply(lambda x: len(x.split())/50)
        df_test['question_per_comment'] = df_test['posts'].apply(lambda x: x.count('?')/50)
        df_test['excl_per_comment'] = df_test['posts'].apply(lambda x: x.count('!')/50)
        df_test['upper_case'] = df_test['posts'].str.findall(r'[A-Z]').str.len()/50
        df_test[['polarity', 'subjectivity']] = df_test['posts'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
        df_test['ellipsis_per_comment'] = df_test['posts'].apply(lambda x: x.count('...')/50)
        Xx = df_test.drop(['posts'], axis=1)
        X_features = scaler.transform(Xx)
        
        predictionIE = modelIE.predict(X_features)
        ur_comment1 = f'Answer1: {input_features[0]}'
        ur_comment2 = f'Answer2: {input_features[1]}'
        ur_comment3 = f'Answer3: {input_features[2]}'
        testIE_result = f'I-E Result: {round(predictionIE[0][0]*100,2)}% Introvert, {round(predictionIE[0][1]*100,2)}% Extrovert' 
        predictionNS = modelNS.predict(X_features)
        testNS_result = f'N-S Result: {round(predictionNS[0][0]*100,2)}% Intuitive, {round(predictionNS[0][1]*100,2)}% Sensing' 

        output = [ur_comment1, ur_comment2, ur_comment3, testIE_result, testNS_result]


        return render_template('prediction.html', prediction_text = output )

if __name__ == "__main__":
    app.run(threaded=True, debug=True, port=5000)


















# # Use PyMongo to establish Mongo connection
# # mongo = PyMongo(app, uri="mongodb://localhost:27017/weather_app")
# mongo = PyMongo(app, uri="mongodb+srv://hhos:Password1@cluster0-2fcii.mongodb.net/weather_data?retryWrites=true&w=majority")
# # mongo = PyMongo(app, uri="mongodb://<dbuser>:<dbpassword>@ds231749.mlab.com:31749/heroku_ft9m418t&retryWrites=False")

# print(mongo)


# # Route to render index.html template using data from Mongo
# @app.route("/livedata")
# def livedata():

#     # Find one record of data from the mongo database
#     xxx_data = mongo.db.weather.find_one()  
#     # Return template and data
#     return render_template("livedata_heroku.html", weather=xxx_data) # 

# @app.route("/")
# def home():
#     return render_template("alldata.html")
                                      
# @app.route("/getmygraph/<path:path>")
# def send_bar(path):
#     print(path)
#     return send_from_directory('templates/', path) 

# @app.route("/getmygraph2/<path:path>")
# def send_bar2(path):
#     print(path)
#     return send_from_directory('templates/', path)

# # @app.route("/getmygraph3/<path:path>")
# # def url(path):
# #     urlxxx = { 'urlicon':'http://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css' }
# #     print(urlxxx['urlicon'])
# #     return send_from_directory( urlxxx['urlicon'], path)


# @app.route("/getmyfiles/<path:path>")
# def send_js(path):
#     print(path)
#     return send_from_directory('templates/static/js/', path)



# # Route that will trigger the scrape function
# @app.route("/scrape")
# def scrape():


#     # Run the scrape function  the function in scrape_mars
#     weather_data = scrape_aqiheroku.scrape()

#     # Update the Mongo database using update and upsert=True  weather is the collection
#     mongo.db.weather.update({}, weather_data, upsert=True)

#     print("Mongo DB updated")

#     # Redirect back to home page
#     return redirect("/")


# if __name__ == "__main__":
#     app.run(debug=True, port = 5000)
