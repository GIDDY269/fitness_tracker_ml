import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
import os
from src.logger import logging
from src.exception import CustomException
import urllib.request
import zipfile



class Data_ingestion:

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


        except Exception as e :
            raise CustomException(e,sys)
        

if __name__ == '__main__' :
    obj = Data_ingestion()
    obj.initiate_data_ingestion()
        


            



