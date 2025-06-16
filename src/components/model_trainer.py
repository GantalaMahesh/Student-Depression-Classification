from src.utils import save_object, evaluate_models
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, classification_report

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                # "Support Vector Machine": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            params = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [1000]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': [None, 'balanced']
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': [None, 'balanced']
                },
                '''
                "Support Vector Machine": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': [None, 'balanced']
                },'''
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "Naive Bayes": {},
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'max_depth': [3, 5, 7]
                }
            }

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, params)

            print("Model Performance Summary:")
            for name, scores in model_report.items():
                print(
                    f"{name}: Recall = {scores['recall']:.4f}, Accuracy = {scores['accuracy']:.4f}, F2 = {scores['f2']:.4f}")

            for name, scores in model_report.items():
                logging.info(
                    f"{name} - Recall: {scores['recall']:.4f}, Accuracy: {scores['accuracy']:.4f}, F2: {scores['f2']:.4f}")

            best_model_name = max(
                model_report.items(),
                key=lambda x: (x[1]["f2"])
            )[0]

            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")

            if model_report[best_model_name]['f2'] < 0.6 or model_report[best_model_name]['recall'] < 0.6:
                raise CustomException(
                    "No suitable model found with acceptable F2-score or recall.")

            logging.info(
                f"Saved best model to {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            report = classification_report(y_test, predicted)

            print("\nClassification Report of Best Model:\n", report)

            logging.info(
                f"Classification Report for {best_model_name}:\n{report}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report[best_model_name]['f2'], best_model

        except Exception as e:
            raise CustomException(e, sys)
