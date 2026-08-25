import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

import time
from p4p.client.thread import Context

ctxt = Context('pva')
base = "ml-nn:ex3:"

def monitor_cb(time_chosen):
    df = pd.read_csv("../may.csv")

    test_data = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    try:   
        start = pd.Timestamp(str(time_chosen).strip().split("'")[1]) 
        input_data = test_data.slice_by_time(start - pd.Timedelta(days=2), start)
        ref_data = test_data.slice_by_time(start, start + pd.Timedelta(days=2)).to_data_frame()
        ref_means = ref_data["target"].values

        predictor = TimeSeriesPredictor.load("/home/joel/Documents/EPICS_COURSE/PROJECT/ML-epics/ex3/src/autogluon-m4-hourly")
        prediction = predictor.predict(input_data)

        df_pred = prediction.to_data_frame()

        means = df_pred["mean"].values
        h24_idx = 23
        h48_idx = 47

        ctxt.put(f"{base}bad", 0)
        ctxt.put(f"{base}temperature", means)
        ctxt.put(f"{base}temperature_ref", ref_means)
        ctxt.put(f"{base}time", np.arange(0,48,1))

        ctxt.put(f"{base}hour_24", means[h24_idx])
        ctxt.put(f"{base}hour_48", means[h48_idx])

        ctxt.put(f"{base}hour_24_q_10", df_pred["0.1"].values[h24_idx])
        ctxt.put(f"{base}hour_24_q_90", df_pred["0.9"].values[h24_idx])

        ctxt.put(f"{base}hour_48_q_10", df_pred["0.1"].values[h48_idx])
        ctxt.put(f"{base}hour_48_q_90", df_pred["0.9"].values[h48_idx])

    except ValueError:
        ctxt.put(f"{base}bad", 1)
        print("Bad Date!")

try: 
    sub = ctxt.monitor(f"{base}set_time", monitor_cb)
    time.sleep(99999)
    sub.close()


except KeyboardInterrupt:
        print("Interupted!")
