import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from utils import LowPassFilter,PrincipalComponentAnalysis,FourierTransformation,NumericalAbstraction
from sklearn.cluster import KMeans
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException



@dataclass
class FeatureEngineerin_config:
    feature_datapath = os.path.join('artifacts','feature_engineered_data.pkl')

class FEATURE_ENGINEERING:
    def __init__(self) :
        self.feature_config = FeatureEngineerin_config()

    def initiate_feature_engineering(self):
        try:
            logging.info('initiating feature engineering')
            data_path = 'artifacts/outlier_removed_chauvenet.pkl'

            df = pd.read_pickle(data_path)

            predicitor_columns = list(df.columns[:6])
    
             # handling missing values using interpolate

            for col in predicitor_columns:
                df[col] = df[col].interpolate()

            logging.info('Applying low pass filter')
            lowpass_df = df.copy()
            sf = 1000/200
            cutoff = 1.3

            lowpass = LowPassFilter()

            
            for col in predicitor_columns:
                lowpass_df = lowpass.low_pass_filter(lowpass_df,sampling_frequency=sf,cutoff_frequency=cutoff,col=col)
                lowpass_df[col] = lowpass_df[col + '_lowpass']
                del lowpass_df[col + '_lowpass']


            logging.info('Applying PCA')

            PCA = PrincipalComponentAnalysis()
            pca_df = lowpass_df.copy()

            pca_df = PCA.apply_pca(pca_df,predicitor_columns,3)


            logging.info('applying sum of squares')

            df_squared = pca_df.copy()

            acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
            gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

            df_squared['acc_r'] = np.sqrt(acc_r)
            df_squared['gyr_r'] = np.sqrt(gyr_r)


            logging.info('Applying rolling average')

            df_temporal = df_squared.copy()
            NumAbs = NumericalAbstraction()

            predicitor_columns = predicitor_columns + ['acc_r','gyr_r']


            ws = int(1000/200)
            df_temporal_list = []
            for s in df_temporal['set'].unique():
                subset = df_temporal[df_temporal['set']==s].copy()

                for col in predicitor_columns:
                    subset = NumAbs.abstract_numerical(subset,[col],ws,'mean')
                    subset = NumAbs.abstract_numerical(subset,[col],ws,'std')
                df_temporal_list.append(subset)

            df_temporal =  pd.concat(df_temporal_list)


            logging.info('Applying fast fourier transformation')

            freq_df = df_temporal.copy().reset_index() # changing it to discrete index not time series
            fft = FourierTransformation()

            sr = int(1000/200)
            ws = int(2800/200) # average length of a repetition

            df_freq_list = []

            for s in freq_df['set'].unique():
                print(f'Applying fourier transformation in set {s}')
                subset = freq_df[freq_df['set'] == s].reset_index(drop=True).copy()
                subset = fft.abstract_frequency(subset,predicitor_columns,ws,sr)
                df_freq_list.append(subset)

            freq_df = pd.concat(df_freq_list).set_index('epoch (ms)',drop=True)

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

            pd.to_pickle(obj=df_cluster,filepath_or_buffer=self.feature_config.feature_datapath)

        except Exception as e:
            raise CustomException(e,sys)



if __name__=='__main__':
    obj = FEATURE_ENGINEERING()
    obj.initiate_feature_engineering()













