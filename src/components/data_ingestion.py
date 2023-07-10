import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import urllib.request
import zipfile
from src.components.data_transformation import Data_transform
from src.components.outlier_removal import outlier_removal

@dataclass
class Data_ingestion_config:
    ingested_data_filepath = os.path.join('artifacts','MetaMotion/*csv')

class Data_ingestion:
    def __init__(self):
        self.data_ingestion_config = Data_ingestion_config()

    def initiate_data_ingestion(self):
        try:
            logging.info('Initiating data ingestion')
            os.makedirs('artifacts',exist_ok=True)
            data_path = 'artifacts/data.zip'
            unzip_path = 'artifacts'
            source_url = 'https://github.com/GIDDY269/human-resource-department/raw/main/MetaMotion.zip'
            if not os.path.exists(data_path): #DOWNLOADING THE DATA
                filename, header = urllib.request.urlretrieve(
                    url = source_url,
                    filename = data_path
                )

                logging.info(f'Downloaded {filename} successfully with the fowwing info \n {header}')

            else:
                logging.info('this file already exist')
            logging.info('Unziping data')
            os.makedirs(unzip_path,exist_ok=True)
            with zipfile.ZipFile(data_path,'r') as zip_rep:
                zip_rep.extractall(unzip_path)
            logging.info('Unzip completed')

            return self.data_ingestion_config.ingested_data_filepath

        except Exception as e :
            raise CustomException(e,sys)
        

if __name__ == '__main__' :
    obj = Data_ingestion()
    path = obj.initiate_data_ingestion()

    trans =  Data_transform()
    path = trans.initiate_data_transform(path)

    out = outlier_removal()
    out.initiate_outlier_removal(path)


            



