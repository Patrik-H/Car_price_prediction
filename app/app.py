from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
loaded_model = joblib.load("model.pkl")
loaded_pipeline = joblib.load("full_pipeline.pkl")

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        return_value = (ValuePredictor(to_predict_list))
        print(return_value)
        return render_template('homepage.html',pred=f'Predicted price: {return_value} eur')
    else:
        return render_template('homepage.html')

def get_dataframe_to_predict(my_list):
    my_columns = ['Tachometer', 'ProductionYear', 'EnginePower', 'Make', 'Body', 'Fuel', 'Gearbox']
    return pd.DataFrame([my_list], columns=my_columns)

def ValuePredictor(my_list):
    dataframe_to_predict = get_dataframe_to_predict(my_list)
    # loaded_model = joblib.load("model.pkl")
    # loaded_pipeline = joblib.load("full_pipeline.pkl")
    data = loaded_pipeline.transform(dataframe_to_predict)
    result = loaded_model.predict(data)
    return result[0]

