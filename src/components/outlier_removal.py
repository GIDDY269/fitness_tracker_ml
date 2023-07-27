import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from utils import mark_outliers_chauvenet


@dataclass
class outlier_removal_config :
    outlier_removed_datapath = os.path.join('artifacts','outlier_removed_chauvenet.pkl')

class outlier_removal:
    def __init__(self):
        self.outlier_filepath = outlier_removal_config()

    def initiate_outlier_removal(self):
        try:
            logging.info('initiating outlier removal using chauvenet')
            # create a for loop to deal with outliers
            data_path = 'artifacts/transformed_data.pkl'
            df = pd.read_pickle(data_path)
            outlier_column = df.columns[:6]
            outliers_remove_df = df.copy()

            logging.info('replacing marked columns as outliers as NaN')

            for col in outlier_column:
                for label in df['label'].unique():
                    dataset = mark_outliers_chauvenet(df[df['label'] == label],col)

                    # replace the columns marked as outliers as NaN
                    dataset.loc[dataset[col+'_outlier'],col] = np.nan

                    #update the column to the original dataframe

                    outliers_remove_df.loc[(outliers_remove_df['label']== label),col] = dataset[col]
                    n_outliers = len(dataset) - len(dataset[col].dropna())
                    print(f'removed {n_outliers} from {col} for {label}') 
            logging.info('outliers removed sucessfully')
            # exporting data        
            pd.to_pickle(obj=outliers_remove_df,filepath_or_buffer=self.outlier_filepath.outlier_removed_datapath)

        
        except Exception as e :
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = outlier_removal()
    obj.initiate_outlier_removal()