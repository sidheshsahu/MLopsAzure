from src.logger import logging
from src.exception import CustomException
from src.utils import model_save
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np



@dataclass
class data_training_config():
    data_trainer:str=os.path.join("artifacts/models",'model.pkl')
    
class TrainingData():
    
    def __init__(self):
        self.data_training=data_training_config()
        
    def initiate_training(self,train_arr,test_arr):
        try:
            X_train =train_arr[:, :-1]
            y_train =train_arr[:, -1]
            X_test=test_arr[:,:-1]
            y_test=test_arr[:, -1]
            
            models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'Random Forest': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier()
                     }
            
            
            params = {
                'Logistic Regression': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear']
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'Random Forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 5, 10]
                },
                'Decision Tree': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10]
                }
               }





            model_report = {}
            best_models = {}

            for name in models:
                
                model = models[name]
                param_grid = params[name]

                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
                gs.fit(X_train, y_train)

                best_model_grid = gs.best_estimator_
               
                y_pred = best_model_grid.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                model_report[name] = acc
                best_models[name] = best_model_grid

                



            sorted_report = dict(sorted(model_report.items(), key=lambda x: x[1], reverse=True))

            best_model_name = next(iter(sorted_report))
            best_model = best_models[best_model_name]
            best_accuracy = sorted_report[best_model_name]

            print("\n Final Model Accuracy Report:")
            for name, acc in sorted_report.items():
                print(f"{name}: {acc:.4f}")

            print(f"\n Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
                        
                        
            logging.info("Get Best Model with better accuracy")   
            
            
            model_save(
                file_path=self.data_training.data_trainer,
                obj=best_model
            )
            

        except Exception as e:
            raise CustomException(e, sys)
