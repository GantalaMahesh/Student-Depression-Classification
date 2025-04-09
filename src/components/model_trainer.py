import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB  

from xgboost import XGBClassifier          
from lightgbm import LGBMClassifier         
from catboost import CatBoostClassifier     
from sklearn.metrics import recall_score, classification_report


from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
           logging.info("split the training and test input data")
           X_train, y_train, X_test, y_test = (
               train_array[:,:-1],
               train_array[:,-1],
               test_array[:,:-1],
               test_array[:,-1]
           )  
          
           

           models = {
               "Logistic Regression": LogisticRegression(),
               "Random Forest": RandomForestClassifier(),
               "Decision Tree": DecisionTreeClassifier(),
               "Support Vector Machine": SVC(),  
               "K-Nearest Neighbors": KNeighborsClassifier(),
               "Naive Bayes": GaussianNB(),
               "AdaBoost": AdaBoostClassifier(),
               "Gradient Boosting": GradientBoostingClassifier(),
               "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
               "LightGBM": LGBMClassifier(),
               "CatBoost": CatBoostClassifier(verbose=0)
           }

           params = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],  # You can also try 'none' or 'elasticnet'
                    'solver': ['liblinear', 'saga'],
                    'class_weight': [None, 'balanced'],
                    'max_iter':[1000]
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
                "Support Vector Machine": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': [None, 'balanced']
                },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "Naive Bayes": {
                    # Usually not many hyperparameters to tune here
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'max_depth': [3, 5, 7]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'scale_pos_weight': [1, 3, 5]  # Useful for imbalanced datasets
                },
                "LightGBM": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'class_weight': [None, 'balanced']
                },
                "CatBoost": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200],
                    'scale_pos_weight': [1, 3, 5]
                }
            }

           
           model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

           # To get the best model score
           best_model_score = max(sorted(model_report.values()))

           # To get best model name
           best_model_name = list(model_report.keys())[
               list(model_report.values()).index(best_model_score)
           ]
           best_model = models[best_model_name]

           logging.info(f"Best individual model: {best_model_name} with recall: {best_model_score}")

           
           if best_model_score<0.6:
               raise CustomException("No suitable model found with acceptable performance.")
           logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
           
           predicted = best_model.predict(X_test)

           recall_score_best_model = recall_score(y_test, predicted)
           report = classification_report(y_test, predicted)

           print("Classification Report:\n", report)

           
           save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

           return recall_score_best_model, best_model
        

        except Exception as e:
            raise CustomException(e, sys)
        
    