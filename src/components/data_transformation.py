import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from glob import glob



@dataclass
class Data_transform_config:
    transformed_datapath = os.path.join('artifacts','transformed_data.pkl')

class Data_transform:
    def __init__(self) :
        self.transform_config = Data_transform_config()

    def initiate_data_transform(self , data_filepath):
        try:

            logging.info('initiating data_transformation')
        
            data_files = glob(data_filepath)

            logging.info(f'The total number of files : {len(data_files)}')

            #extracting features from metamotion datafilepath (participants,category,labels)

            acc_df = pd.DataFrame()
            gyr_df = pd.DataFrame()

            acc_set = 1
            gyr_set = 1

            for f in data_files:
                participants = f.split('-')[0].lstrip('../../artifacts/MetaMotion/')
                category = f.split('-')[2].rstrip('123').rstrip('_MetaWear_2019') 
                label = f.split('-')[1]

                df = pd.read_csv(f)

                df['participants'] = participants
                df['category'] = category
                df['label'] = label


                if 'Accelerometer' in f :
                    df['set'] = acc_set
                    acc_set += 1
                    acc_df = pd.concat([acc_df,df])

                if 'Gyroscope' in f:
                    df['set'] = gyr_set
                    gyr_set += 1
                    gyr_df = pd.concat([gyr_df,df])
                


            # converting time columns to datetime
            acc_df.index = pd.to_datetime(acc_df['epoch (ms)']) # converting unix time to datetime and seting it as index
            gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'])

            logging.info('created accelerometer and gyroscope dataframes')   
                # deleting time,epoch and elaspe column
            del_columns = ['elapsed (s)','time (01:00)','epoch (ms)']

            for col in del_columns:
                del gyr_df[col]  
                del acc_df[col]

            data_merge = pd.concat([acc_df.iloc[:,0:3],gyr_df],axis=1)

            data_merge.columns = [
                'acc_x',
                'acc_y',
                'acc_z',
                'gyr_x',
                'gyr_y',
                'gyr_z',
                'participants',
                'category',
                'label',
                'set'
            ]

            #resampling the dataframe to fillna values due to the fact that the gyroscope sensor was measuring at a high frequency compared to the accelorometer

            sampling = {
                'acc_x': 'mean',
                'acc_y': 'mean',
                'acc_z': 'mean',
                'gyr_x': 'mean',
                'gyr_y': 'mean',
                'gyr_z': 'mean',
                'participants': 'last',
                'category': 'last',
                'label' : 'last',
                'set' : 'last'
            }

            days = [g for n,g in data_merge.groupby(pd.Grouper(freq='D'))] #splitting the dataframe by day

            data_resample = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days]) #resampling it for each day
            logging.info('resampling dataframe')

            data_resample['set'] = data_resample['set'].astype('int')
            data_resample.to_pickle('artifacts/transformed_data.pkl')

            return self.transform_config.transformed_datapath
        
        except Exception as e :
            raise CustomException(e,sys)
