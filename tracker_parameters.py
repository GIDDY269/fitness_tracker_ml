from pydantic import BaseModel
from fastapi import UploadFile,File
import pandas as pd


class SensorData(BaseModel):

    gyroscope :  UploadFile = File(..., description= 'Gyroscope csv file')
    accelerometer : UploadFile = File(..., description= 'Accelerometer cvs file')
    

    
