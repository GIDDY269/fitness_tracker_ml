from fastapi import FastAPI,UploadFile,File
import numpy as np
import pandas as pd
import sys
sys.path.append(r'C:\Users\user\FITNESS_TRACKER')
from tracker_parameters import SensorData
from src.pipeline.predict_pipeline import predict
from joblib import load
from src.logger import logging
import uvicorn
from scipy import stats

app = FastAPI()
pred = predict()

@app.get('/')
def welcome():
    return{'welcome' : 'fitness tracker'}

@app.post('/predict/')
def predict_sensor_data(gyroscope :  UploadFile = File(..., description= 'Gyroscope csv file')
                        ,accelerometer : UploadFile = File(..., description= 'Accelerometer cvs file')):
    
    logging.info('initiating predictions')

    # load model
    model = load('C:/Users/user/FITNESS_TRACKER/artifacts/fitness_tracker.joblib')
    logging.info('loaded model')


    # read csv files
    gyr_df  = pd.read_csv(gyroscope.file)
    acc_df = pd.read_csv(accelerometer.file)

    # precess data

    predict_df = pred.predict_preprocess(gyr_df,acc_df)

    # make predictions

    prediction = model.predict(predict_df)

    if prediction.max() == 'bench' :
        excercise = 'bench press'
    elif prediction.max() == 'ohp':
        excercise = 'over head press'
    elif prediction.max() == 'dead':
        excercise = 'dead lift'
    elif prediction.max() == 'squat' :
        excercise = 'squat'
    elif prediction.max() == 'row' :
        excercise = 'Row'
    else :
        excercise = 'resting'

    if excercise != 'resting' :
        return {"predictions": f'Heyyya , it seems you are doing a {excercise}'}
    else :
        return { 'prediction' : f'Hey , seems you are {excercise}'}


if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8080)

