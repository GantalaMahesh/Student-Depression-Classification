import os
import sys
import dill

from src.exception import CustomException
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            para = param.get(model_name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            recall = recall_score(y_test, y_test_pred)
            accuracy = accuracy_score(y_test, y_test_pred)

            report[model_name] = {
                "recall": recall,
                "accuracy": accuracy
            }

            print(f"{model_name} ✅ Recall: {recall:.4f}, Accuracy: {accuracy:.4f}\n")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
