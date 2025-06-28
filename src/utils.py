from src.logger import logging
from src.exception import CustomException
import pickle
import os
import sys



def save_object(file_path,obj):
        try:
        
                dir_path = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)

                with open(file_path, 'wb') as file_obj:
                    pickle.dump(obj, file_obj)

                logging.info(f"Object saved successfully at: {file_path}")

        except Exception as e:
            
            raise Exception(f"Error saving object to {file_path}: {e}")
        
        
def model_save(file_path,obj):
        try:
        
                dir_path = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)

                with open(file_path, 'wb') as file_obj:
                    pickle.dump(obj, file_obj)

                logging.info(f"Object saved successfully at: {file_path}")

        except Exception as e:
            
            raise Exception(f"Error saving object to {file_path}: {e}")


def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)     