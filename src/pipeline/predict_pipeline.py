import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import load_object
import pickle
import pandas as pd
import numpy as np



class CustomData:
    def __init__(self, feature1, feature2, feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10):  
        self.CGPA = feature1
        self.Internships = feature2
        self.Projects = feature3
        self.Workshops_Certifications = feature4
        self.AptitudeTestScore = feature5
        self.SoftSkillsRating = feature6
        self.ExtracurricularActivities = feature7
        self.PlacementTraining = feature8
        self.SSC_Marks = feature9
        self.HSC_Marks = feature10
        

    def to_dataframe(self):
        try:
            data = {
            "CGPA": [self.CGPA],
            "Internships": [self.Internships],
            "Projects": [self.Projects],
            "Workshops/Certifications": [self.Workshops_Certifications],
            "AptitudeTestScore": [self.AptitudeTestScore],
            "SoftSkillsRating": [self.SoftSkillsRating],
            "ExtracurricularActivities": [self.ExtracurricularActivities],
            "PlacementTraining": [self.PlacementTraining],
            "SSC_Marks": [self.SSC_Marks],
            "HSC_Marks": [self.HSC_Marks]
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features_df):
        try:
            model_path = "artifacts/models/model.pkl"
            preprocessor_path = "artifacts/models/preprocessing.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            processed_data = preprocessor.transform(features_df)
            prediction = model.predict(processed_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)