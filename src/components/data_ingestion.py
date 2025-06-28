import os
# from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_training import TrainingData


@dataclass
class data_ingestion_config():
    train_data_path:str=os.path.join('artifacts/data','train.csv')
    test_data_path:str=os.path.join('artifacts/data','test.csv')
    raw_data_path:str=os.path.join('artifacts/data','raw.csv')

class data_ingestion():
    def __init__(self):
        self.data_ingestion_data=data_ingestion_config()  
        
    def initiate_ingestion(self):
        try:
                
            df = pd.read_csv("notebooks/placementdata.csv")
            os.makedirs(os.path.dirname(self.data_ingestion_data.raw_data_path), exist_ok=True)

                
            df.to_csv(self.data_ingestion_data.raw_data_path, index=False)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.data_ingestion_data.train_data_path, index=False)
            test_df.to_csv(self.data_ingestion_data.test_data_path, index=False)

            # logging.info("Data ingestion completed successfully.")
            return (
                    self.data_ingestion_data.train_data_path,
                    self.data_ingestion_data.test_data_path
                )
        
        except Exception as e:
           
            raise CustomException(e, sys)
         


if __name__=="__main__":
    obj_injest=data_ingestion()
    train_data,test_data=obj_injest.initiate_data_ingestion()
    print("DataUpdated")
    
    obj_transform=DataTransformation()
    train_arr,test_arr=obj_transform.initiate_data_transformation(train_data,test_data)
    
    obj_training=TrainingData()
    obj_training.initiate_training(train_arr,test_arr)
    print("All pipeline ran successfully")