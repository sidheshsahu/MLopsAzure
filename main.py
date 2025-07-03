from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import DataTransformation
from src.components.data_training import TrainingData


if __name__== "__main__":
        obj_ingest=data_ingestion()
        train_data,test_data=obj_ingest.initiate_ingestion()
        print("Data Ingested")
        
        obj_transform=DataTransformation()
        train_arr,test_arr=obj_transform.initiate_data_transformation(train_data,test_data)
        print("Data Transformed")
        
        obj_training=TrainingData()
        obj_training.initiate_training(train_arr,test_arr)
        print("Data Trained")