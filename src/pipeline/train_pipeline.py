"""
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys


def run_pipeline():
    try:
        logging.info("🚀 Starting training pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(
            train_path, test_path)

        # Step 3: Model Training
        trainer = ModelTrainer()
        trainer.initiate_model_training(train_arr, test_arr)

        logging.info("✅ Training pipeline completed successfully!")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()
"""
