from pydantic import BaseModel
from datetime  import datetime

class tracker_params(BaseModel):
    timestamp : datetime
    acc_x : float
    acc_y : float
    acc_z : float
    gyr_x : float
    gyr_y : float
    gyr_z : float