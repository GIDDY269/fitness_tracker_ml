from src.components.data_ingestion import Data_ingestion
from src.components.data_transformation import Data_transform
from src.components.outlier_removal import outlier_removal
from src.components.build_features import FEATURE_ENGINEERING



stage_name = 'DATA INGESTION'

print('Current stage : {stage_name}')
# read in data from database
ingestion = Data_ingestion()

ingested_data = ingestion.initiate_data_ingestion()


stage_name = 'DATA TRANSFORMATION AND PROCESSING'

print('Current stage : {stage_name}')
# transform data and preprocessing

preprocess = Data_transform()
transfromed_data = preprocess.initiate_data_transform(ingested_data)


stage_name = 'OUTLIER REMOVAL'

print('Current stage : {stage_name}')
# removing outliers
outlier = outlier_removal()
outliers_removed_data = outlier.initiate_outlier_removal(transfromed_data)


stage_name = 'FEATURE ENGINEERING'

print('Current stage : {stage_name}')
# perform feature engineering

feature = FEATURE_ENGINEERING()
featured_data = feature.initiate_feature_engineering(outliers_removed_data)





