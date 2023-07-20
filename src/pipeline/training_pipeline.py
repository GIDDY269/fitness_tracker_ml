from src.components.data_ingestion import Data_ingestion
from src.components.data_transformation import Data_transform
from src.components.outlier_removal import outlier_removal
from src.components.build_features import FEATURE_ENGINEERING

# read in data from database
ingestion = Data_ingestion()

ingested_data = ingestion.initiate_data_ingestion()

# transform data and preprocessing

preprocess = Data_transform()
transfromed_data = preprocess.initiate_data_transform(ingested_data)

# removing outliers
outlier = outlier_removal()
outliers_removed_data = outlier.initiate_outlier_removal(transfromed_data)

# perform feature engineering

feature = FEATURE_ENGINEERING()
featured_data = feature.initiate_feature_engineering(outliers_removed_data)




