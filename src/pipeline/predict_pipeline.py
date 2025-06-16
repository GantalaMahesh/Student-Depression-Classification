import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    Handles loading of pre-trained model and preprocessing steps,
    and performs prediction on new input data.
    """

    def __init__(self):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            self.model = load_object(file_path=model_path)
            preprocessing_bundle = load_object(file_path=preprocessor_path)

            self.target_encoder = preprocessing_bundle["target_encoder"]
            self.preprocessor = preprocessing_bundle["preprocessor"]

        except Exception as e:
            raise CustomException(
                f"Error loading model or preprocessor: {e}", sys)

    def predict(self, features: pd.DataFrame):
        try:
            # ✅ Fill missing values
            features = features.fillna("Unknown")

            # ✅ Cast to string for encoding
            features[["Degree", "Profession"]] = features[[
                "Degree", "Profession"]].astype(str)

            # ✅ Handle unknown categories by using try-except or custom fallback
            try:
                features[["Degree", "Profession"]] = self.target_encoder.transform(
                    features[["Degree", "Profession"]]
                )
            except Exception as e:
                # Fallback: replace unknowns with a neutral value like the mean
                for col in ["Degree", "Profession"]:
                    known_categories = self.target_encoder.mapping_[
                        col].dropna()
                    default_value = known_categories.mean()
                    features[col] = features[col].map(
                        self.target_encoder.mapping_[col]
                    ).fillna(default_value)

            # ✅ Apply preprocessing
            transformed_features = self.preprocessor.transform(features)

            # ✅ Predict class and probability
            prediction = self.model.predict(transformed_features)
            probability = self.model.predict_proba(transformed_features)[0][1]

            return prediction[0], probability

        except Exception as e:
            raise CustomException(f"Prediction failed: {e}", sys)


class CustomData:
    """
    Converts raw user inputs into a pandas DataFrame suitable for model prediction.
    """

    def __init__(
        self,
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
        Family_History_of_Mental_Illness: str,
    ):
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

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Creates a DataFrame from the input data.

        Returns:
            pd.DataFrame: Single-row DataFrame for prediction.
        """
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
                "Family History of Mental Illness": [self.Family_History_of_Mental_Illness],
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(
                f"Error creating DataFrame from user input: {e}", sys)
