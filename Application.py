import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home_page.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])

def predict():
    data=[float(x) for x in request.form.values()]
    final_data=scaler.transform(np.array(data).reshape(1,-1))
    final_output=model.predict(final_data)
    return render_template("home_page.html",prediction_text="The House price prdiction is {}".format(final_output))

if __name__=="__main__":
    app.run(debug=True)