from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class DataTransformationConfig():
    preprocessing_obj_pickle:str=os.path.join("artifacts/models","preprocessing.pkl")
    
class DataTransformation():
    def __init__(self):
        self.preprocessing_data=DataTransformationConfig()
        
    def get_transformation_preprocess(self):
        try:
            numerical_columns=['CGPA','Internships','Projects','Workshops/Certifications','AptitudeTestScore','SoftSkillsRating','SSC_Marks','HSC_Marks']
            categorical_columns=['ExtracurricularActivities','PlacementTraining']
            
            numerical_pipeline=Pipeline(
            
            steps=[
                ("impute",SimpleImputer(strategy='mean')),
                ("scalar",StandardScaler())
                ] 
            )

            categorical_pipeline=Pipeline(
            steps=[
            ("impute",SimpleImputer(strategy='most_frequent')),
            ("labelEncoder",OneHotEncoder()),
                ]
            )
            
            transformer=ColumnTransformer(
                transformers=[
                        ('num', numerical_pipeline,numerical_columns),
                        ('cat', categorical_pipeline, categorical_columns)
                    ],
            )
            
            
            return transformer
            
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preprocessing_part=self.get_transformation_preprocess()
            
            
            le=LabelEncoder()
            
            train_df['PlacementStatus']=le.fit_transform(train_df['PlacementStatus'])
            test_df['PlacementStatus']=le.transform(test_df['PlacementStatus'])
            
            target_feature_train_df=train_df['PlacementStatus']
            target_feature_test_df=test_df['PlacementStatus']
            
            train_df=train_df.drop('PlacementStatus',axis=1)
            test_df=test_df.drop('PlacementStatus',axis=1)
            
            
            train_preprocessed_data=preprocessing_part.fit_transform(train_df)
            test_preprocessed_data=preprocessing_part.transform(test_df)
            
            logging.info("Preprocessed Data Successfully")
            
            train_arr=np.c_[
                train_preprocessed_data, np.array(target_feature_train_df)
            ]
            test_arr=np.c_[test_preprocessed_data, np.array(target_feature_test_df)]

            
            save_object(
                file_path=self.preprocessing_data.preprocessing_obj_pickle,
                obj=preprocessing_part
            )
            
            return(
                train_arr,
                test_arr
            )
            
            
            
        except Exception as e:
            raise CustomException(e, sys)
            
          
        
    