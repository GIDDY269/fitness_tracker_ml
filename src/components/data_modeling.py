import  pandas as pd
import numpy as np
import os
import sys 
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from joblib import dump
from sklearn.model_selection import train_test_split
from utils import ClassificationAlgorithms
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
import json



@dataclass
class model_trainer_config:
    model_path = os.path.join('artifacts','fitness_tracker.joblib')

class model_trainer:

    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_trainer(self):

        try:
            logging.info('initiating model trainer')

            datapath = 'artifacts/feature_engineered_data.pkl'

            learner = ClassificationAlgorithms()

            df = pd.read_pickle(datapath)
            df_train = df.drop(['participants','category','set'],axis = 1)

            x = df_train.drop('label',axis=1)
            y = df_train['label']

            x_train,x_test,y_train,y_test = train_test_split(
                                            x,y,test_size=0.25,random_state=0,stratify=y
                                                        )
            
            iterations = 4
            performance_acc = 0
            
            logging.info('Training model')
            for it in range(0,iterations):
                print('\t Training random forest classifier',it)


                (
                    class_train_y,
                    class_test_y,
                    class_train_prob_y,
                    class_test_prob_y,
                    rf
                ) = learner.random_forest(
                                x_train,y_train,x_test,gridsearch=True,print_model_details=True
                                        )
                
                performance_acc += accuracy_score(y_test,class_test_y)

            logging.info('evaluating model')
            performance_acc = performance_acc/iterations                                 

            model_metrics = {
                'accuracy' : performance_acc,
            }

            with open('model_metrics.json' , 'w') as jsonfile:
                json.dump(model_metrics,jsonfile)

            #saving model
            dump(rf,self.model_trainer_config.model_path)

            logging.info('model training completed')
            
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = model_trainer()
    obj.initiate_model_trainer()