import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'Age',
                'Academic Pressure',
                'Work Pressure',
                'CGPA',
                'Study Satisfaction',
                'Job Satisfaction',
                'Work/Study Hours',
                'Financial Stress'
            ]
            
            cat_nominal_features = ['Gender']
            cat_ordinal_features = ["Sleep Duration", "Dietary Habits"]
            binary_features = ["Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]

            # Pipelines
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_nominal_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )

            cat_ordinal_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder()),
                    ('scaler', StandardScaler())
                ]
            )

            cat_binary_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('binary_encoder', OrdinalEncoder())
                ]
            )

            logging.info("Numerical, Nominal, Ordinal, and Binary pipelines set up.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_nominal_pipeline', cat_nominal_pipeline, cat_nominal_features),
                    ('cat_ordinal_pipeline', cat_ordinal_pipeline, cat_ordinal_features),
                    ('cat_binary_pipeline', cat_binary_pipeline, binary_features)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded.")

            target_column_name = 'Depression'
            cat_target_encoding_features = ['Degree', 'Profession']

            # Apply Target Encoding before ColumnTransformer
            target_encoder = ce.TargetEncoder()
            train_df[cat_target_encoding_features] = target_encoder.fit_transform(
                train_df[cat_target_encoding_features], train_df[target_column_name]
            )
            test_df[cat_target_encoding_features] = target_encoder.transform(
                test_df[cat_target_encoding_features]
            )

            logging.info("Target encoding applied to Degree and Profession columns.")

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Fitting preprocessor on training data.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
