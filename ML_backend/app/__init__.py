from flask import Flask,render_template,request, jsonify
from flask_cors import CORS
import json
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sklearn 
from sklearn.linear_model import LinearRegression
import csv


app = Flask(__name__)
CORS(app)
print("CORS*************")
app.secret_key = "YOUR-SECRET-KEY"


head=["state_name","region_name","percent_qualified","yearly_sunlight_kwh_total","number_of_panels_total","lat_avg","lng_avg","kw_total"]


loaded_model1 = joblib.load('app/Regression_model_for_Power_Generation.pkl')
loaded_model2 = joblib.load('app/TS_model_for_Efficency_Year_month.pkl')

def setScale():
    data = pd.read_csv('C:/Users/Hansaka/Desktop/backend/data.csv')
    data=data.drop(labels=['yearly_sunlight_kwh_e', 'yearly_sunlight_kwh_n','yearly_sunlight_kwh_s','yearly_sunlight_kwh_w','number_of_panels_e','number_of_panels_s','number_of_panels_w','yearly_sunlight_kwh_f','number_of_panels_f','number_of_panels_n'], axis=1)
    data=data.dropna()
    country_encoder = LabelEncoder()
    country_encoder.fit(data['state_name'])
    country_code = country_encoder.transform(data['state_name'])
    X = data.iloc[:,:-1]
    X['Country_code']=country_code.tolist()
    X=X.drop(columns=['state_name'],axis=1)
    # Fit the Scaler
    scaler = StandardScaler().fit(X)
    return scaler 

from sklearn.metrics import mean_squared_error


def writeData(header,data):
    with open('app/data.csv', 'w',newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        print(header)
        print(data)
        writer.writerow(header)
        # write the data
        writer.writerow(data)


@app.route("/test")
def test():
    print("*******************")
    return {"Status":True,"data":23333}, 200

@app.route('/predict_api',methods=['POST'])
def predict_api():
    # try:       
        model_nuber = request.json.get('model')
        if model_nuber == 1:
            scaler = setScale()
            model_data = request.json.get('data')
            writeData(head,model_data)
            data_test = pd.read_csv('app/data.csv')
            data_test=data_test.drop(labels=['yearly_sunlight_kwh_e', 'yearly_sunlight_kwh_n','yearly_sunlight_kwh_s','yearly_sunlight_kwh_w','number_of_panels_e','number_of_panels_s','number_of_panels_w','yearly_sunlight_kwh_f','number_of_panels_f','number_of_panels_n'], axis=1, errors='ignore')
            print(data_test)
            X_test = data_test.iloc[:,:-1]
            Y_test = data_test.iloc[:,-1]
            country_encoder_test = LabelEncoder()
            country_encoder_test.fit(data_test['state_name'])
            country_code_test = country_encoder_test.transform(data_test['state_name'])
            X_test['Country_code']=country_code_test.tolist()
            X_test=X_test.drop(columns=['state_name'],axis=1)
            X_scaled= scaler.transform(X_test.iloc[[0]])
            print(X_scaled)

            Y_pred=loaded_model1.predict(X_scaled)
            output = json.dumps(Y_pred.tolist())
            print(output)
            return output
        elif model_nuber == 2:
            startDate = request.json.get('start')
            endDate = request.json.get('end')
            res=loaded_model2.predict(startDate,endDate)
            print(startDate)
            print(endDate)
            return res.to_json()
        else:
                return jsonify({"status":"Model not found! Please rebuild"})


@app.route("/")
def hello():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=5000,host='0.0.0.0', )