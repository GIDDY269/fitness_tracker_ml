import os
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from src.exception import CustomException
from src.components.data_ingestion import Data_ingestion
from src.components.data_transformation import Data_transform
from src.components.outlier_removal import outlier_removal
from src.components.build_features import FEATURE_ENGINEERING
from src.components.data_modeling import model_trainer


try:
    stage_name = 'DATA INGESTION'

    print(f'Current stage : {stage_name}')
    # read in data from database
    ingestion = Data_ingestion()

    ingested_data = ingestion.initiate_data_ingestion()


    stage_name = 'DATA TRANSFORMATION AND PROCESSING'

    print(f'Current stage : {stage_name}')
    # transform data and preprocessing

    preprocess = Data_transform()
    transfromed_data = preprocess.initiate_data_transform()


    stage_name = 'OUTLIER REMOVAL'

    print(f'Current stage : {stage_name}')
    # removing outliers
    outlier = outlier_removal()
    outliers_removed_data = outlier.initiate_outlier_removal()


    stage_name = 'FEATURE ENGINEERING'

    print(f'Current stage : {stage_name}')
    # perform feature engineering

    feature = FEATURE_ENGINEERING()
    featured_data = feature.initiate_feature_engineering()


    stage_name = 'DATA MODELLING'

    print(f'Current stage : {stage_name}')

    model = model_trainer()

    model.initiate_model_trainer()
except Exception as e:
    raise CustomException(e,sys)



