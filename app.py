from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                Gender=request.form.get('Gender'),
                Age=float(request.form.get('Age')),
                Profession=request.form.get('Profession'),
                Academic_Pressure=float(request.form.get('Academic_Pressure')),
                Work_Pressure=float(request.form.get('Work_Pressure')),
                CGPA=float(request.form.get('CGPA')),
                Study_Satisfaction=float(
                    request.form.get('Study_Satisfaction')),
                Job_Satisfaction=float(request.form.get('Job_Satisfaction')),
                Sleep_Duration=request.form.get('Sleep_Duration'),
                Dietary_Habits=request.form.get('Dietary_Habits'),
                Degree=request.form.get('Degree'),
                Have_you_ever_had_suicidal_thoughts=request.form.get(
                    'Have_you_ever_had_suicidal_thoughts'),
                Work_Study_Hours=float(request.form.get('Work_Study_Hours')),
                Financial_Stress=float(request.form.get('Financial_Stress')),
                Family_History_of_Mental_Illness=request.form.get(
                    'Family_History_of_Mental_Illness')
            )

            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=int(results[0]))

        except Exception as e:
            print("Prediction Error:", str(e))
            return render_template('home.html', results="Prediction Error")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
