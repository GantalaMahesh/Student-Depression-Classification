import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
           df = pd.read_csv("notebook/data/student_depression_dataset.csv")
           logging.info('Read the Dataset as Dataframe')

           df.drop(columns=['id', 'City'], inplace=True)

           df.loc[:, 'Financial Stress'] = df['Financial Stress'].replace('?', np.nan)
           df.loc[:, 'Financial Stress'] = df['Financial Stress'].fillna(df['Financial Stress'].mode()[0])
           df.loc[:, 'Financial Stress'] = df['Financial Stress'].astype(float)
           
           artifacts_dir = os.path.dirname(self.ingestion_config.train_data_path)
           if not os.path.exists(artifacts_dir):
               os.makedirs(artifacts_dir, exist_ok=True)

           df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

           logging.info("Train Test split Initiated")
           train_set, test_set =  train_test_split(df, test_size=0.2, random_state=42)

           train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

           test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

           logging.info('Ingestion of the Data is completed')

           return(
               self.ingestion_config.train_data_path,
               self.ingestion_config.test_data_path
           )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
    