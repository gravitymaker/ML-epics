import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

import time
from p4p.client.thread import Context
    

time_chosen = "2026-05-15 00:12:30"

df = pd.read_csv("../may.csv")

test_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)

end = pd.Timestamp(time_chosen)
start = end - pd.Timedelta(days=2)

input_data = test_data.slice_by_time(start, end)

predictor = TimeSeriesPredictor.load("/home/joel/Documents/EPICS_COURSE/PROJECT/ML-epics/ex3/src/autogluon-m4-hourly")
prediction = predictor.predict(input_data)

df_pred = prediction.to_data_frame()

print(df_pred)