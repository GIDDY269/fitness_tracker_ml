import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from src.exception import CustomException
from src.logger import logging
from utils import LowPassFilter,PrincipalComponentAnalysis,FourierTransformation,NumericalAbstraction
from sklearn.cluster import KMeans




class predict :

    def predict_preprocess(self,gyr_df:pd.DataFrame, acc_df:pd.DataFrame):
        try:

            # setting epoch as index

            gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'],unit='ms')
            acc_df.index = pd.to_datetime(acc_df['epoch (ms)'],unit='ms')

            del_columns = ['elapsed (s)','time (01:00)','epoch (ms)']

            for col in del_columns:
                del gyr_df[col]  
                del acc_df[col]
            
            logging.info('combining dataframe')
            data_merge = pd.concat([acc_df.iloc[:,:3],gyr_df],axis=1)

            data_merge.columns = [
                'acc_x',
                'acc_y',
                'acc_z',
                'gyr_x',
                'gyr_y',
                'gyr_z'
            ]

            sampling = {
                'acc_x': 'mean',
                'acc_y': 'mean',
                'acc_z': 'mean',
                'gyr_x': 'mean',
                'gyr_y': 'mean',
                'gyr_z': 'mean'
            }

            daily_data = data_merge.groupby(pd.Grouper(freq='D'))
            data_resample = pd.DataFrame()
            for day, day_data in daily_data:
                resample_day_data = day_data.resample(rule='200ms').apply(sampling).dropna()
                logging.info(resample_day_data)
                data_resample = pd.concat([data_resample, resample_day_data])

            # handling missing values using interpolate
            for col in data_resample.columns:
                data_resample[col] = data_resample[col].interpolate()


            logging.info('Applying low pass filter in prediction pipeline')
            predicitor_columns = list(data_resample.columns)
            lowpass_df = data_resample.copy()
            sf = 1000/200
            cutoff = 1.3

            lowpass = LowPassFilter()

            
            for col in predicitor_columns:
                lowpass_df = lowpass.low_pass_filter(lowpass_df,sampling_frequency=sf,cutoff_frequency=cutoff,col=col)
                lowpass_df[col] = lowpass_df[col + '_lowpass']
                del lowpass_df[col + '_lowpass']
            logging.info(f'low pass df columns {lowpass_df.columns}')
            
            logging.info('Applying PCA in predict pipeline')

            PCA = PrincipalComponentAnalysis()
            pca_df = lowpass_df.copy()

            pca_df = PCA.apply_pca(pca_df,predicitor_columns,3)


            logging.info('applying sum of squares in predict pipeline')

            df_squared = pca_df.copy()

            acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
            gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

            df_squared['acc_r'] = np.sqrt(acc_r)
            df_squared['gyr_r'] = np.sqrt(gyr_r)

            logging.info('Applying rolling average in predict pipeline')

            df_temporal = df_squared.copy()
            NumAbs = NumericalAbstraction()

            predicitor_columns = predicitor_columns + ['acc_r','gyr_r']


            ws = int(1000/200)
            df_temporal_list = []

            for col in predicitor_columns:
                df_temporal = NumAbs.abstract_numerical(df_temporal,[col],ws,'mean')
                df_temporal = NumAbs.abstract_numerical(df_temporal,[col],ws,'std')
            df_temporal_list.append(df_temporal)

            df_temporal =  pd.concat(df_temporal_list)
            logging.info(f'columns of temporal(rolling average) df {df_temporal.columns}')

            logging.info('Applying fast fourier transformation')

            freq_df = df_temporal.copy().reset_index() # changing it to discrete index not time series
            fft = FourierTransformation()

            sr = int(1000/200)
            ws = int(2800/200) # average length of a repetition


            print(f'Applying fourier transformation')
            freq_df = freq_df.reset_index(drop=True).copy()
            freq_df = fft.abstract_frequency(freq_df,predicitor_columns,ws,sr)

            freq_df = freq_df.set_index('epoch (ms)',drop=True)

            logging.info(f' columna after applying fourier transform {freq_df.columns}')

            logging.info('allowing 50% correlation')

            
            freq_df.dropna(inplace=True) #drop missing values
            freq_df = freq_df[::2]

            logging.info('Adding clusters')

            df_cluster = freq_df.copy()

            cluster_columns = ['acc_x','acc_y','acc_z']

            subset = df_cluster[cluster_columns]    
            kmeans = KMeans(n_clusters=5,n_init=20,random_state=0)
            df_cluster['cluster'] = kmeans.fit_predict(subset)

            logging.info(f'columns of data {list(df_cluster.columns)}')

            return df_cluster
        except Exception as e:
            raise CustomException(e,sys)





              




