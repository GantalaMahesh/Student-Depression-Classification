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
           
           model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

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
        
    