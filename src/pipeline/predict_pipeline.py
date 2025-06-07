"""
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessing_bundle = load_object(file_path=preprocessor_path)
            # Extract target_encoder and preprocessor
            target_encoder = preprocessing_bundle["target_encoder"]
            preprocessor = preprocessing_bundle["preprocessor"]

            # Apply target encoding on Degree and Profession before transformation
            features[["Degree", "Profession"]] = target_encoder.transform(
                features[["Degree", "Profession"]])

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Gender: str,
                 Age: int,
                 Profession: str,
                 Academic_Pressure: int,
                 Work_Pressure: int,
                 CGPA: float,
                 Study_Satisfaction: int,
                 Job_Satisfaction: int,
                 Sleep_Duration: str,
                 Dietary_Habits: str,
                 Degree: str,
                 Have_you_ever_had_suicidal_thoughts: str,
                 Work_Study_Hours: int,
                 Financial_Stress: int,
                 Family_History_of_Mental_Illness: str):

        self.Gender = Gender
        self.Age = Age
        self.Profession = Profession
        self.Academic_Pressure = Academic_Pressure
        self.Work_Pressure = Work_Pressure
        self.CGPA = CGPA
        self.Study_Satisfaction = Study_Satisfaction
        self.Job_Satisfaction = Job_Satisfaction
        self.Sleep_Duration = Sleep_Duration
        self.Dietary_Habits = Dietary_Habits
        self.Degree = Degree
        self.Have_you_ever_had_suicidal_thoughts = Have_you_ever_had_suicidal_thoughts
        self.Work_Study_Hours = Work_Study_Hours
        self.Financial_Stress = Financial_Stress
        self.Family_History_of_Mental_Illness = Family_History_of_Mental_Illness

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Profession": [self.Profession],
                "Academic Pressure": [self.Academic_Pressure],
                "Work Pressure": [self.Work_Pressure],
                "CGPA": [self.CGPA],
                "Study Satisfaction": [self.Study_Satisfaction],
                "Job Satisfaction": [self.Job_Satisfaction],
                "Sleep Duration": [self.Sleep_Duration],
                "Dietary Habits": [self.Dietary_Habits],
                "Degree": [self.Degree],
                "Have you ever had suicidal thoughts ?": [self.Have_you_ever_had_suicidal_thoughts],
                "Work/Study Hours": [self.Work_Study_Hours],
                "Financial Stress": [self.Financial_Stress],
                "Family History of Mental Illness": [self.Family_History_of_Mental_Illness]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
"""
