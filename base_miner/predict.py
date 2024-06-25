# developer: Foundry Digital
# Copyright © 2023 Foundry Digital

# Import modules that already exist or can be installed using pip
from datetime import datetime, timedelta
from threading import Thread
from base_miner.model import retrain_and_save
import json
import joblib
import numpy as np
import pandas as pd
from pytz import timezone
from sklearn.preprocessing import MinMaxScaler
# from base_miner.model import create_and_save_base_model_lstm, create_and_save_base_model_regression

# import custom defined files
from base_miner.get_data import load_df, save_df, merge_dfs, prep_data, scale_data, round_down_time


def save_model_retraining_args(X_scaled: np.ndarray, y_scaled: np.ndarray):
    scaled_vars = {
        "X_scaled": X_scaled.tolist(),
        "y_scaled": y_scaled.tolist()
    }
    with open("scaled_vars.json", "w") as file:
        json.dump(scaled_vars, file, indent=4)

    print("Successfully saved scaled vars.")


def predict(timestamp: str, model, scaler: MinMaxScaler or None = None, X_scaled: np.ndarray or None = None,
            y_scaled: np.ndarray or None = None,
            type="arimax") -> float:
    """
    Predicts the close price of the next 5m interval

    The predict function also ensures that the data is procured - using yahoo finance's python module,
    prepares the data and gets basic technical analysis metrics, and finally predicts the model
    and scales it based on the scaler used previously to create the model.

    Input:
        :param timestamp: The timestamp of the instant at which the request is sent by the validator
        :type timestamp: datetime.datetime

        :param scaler: The scaler used to scale the inputs during model training process
        :type scaler: sklearn.preprocessing.MinMaxScaler

        :param model: The model used to make the predictions - in this case a .h5 file
        :type model: A keras model instance

    Output:
        :returns: The close price of the 5m period that ends at the timestamp passed by the validator
        :rtype: float
    """
    # calling this to get the data - the information passed by the validator contains
    # only a timestamp, it is on the miners to get the data and prepare is according to their requirements
    data = prep_data(drop_na=False)
    # Ensuring that the Datetime column in the data procured from yahoo finance is truly a datetime object
    if type == "lstm":
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data['Datetime'] = data['Datetime'].dt.tz_convert("America/New_York")
        data[['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']] = data[
            ['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']].interpolate(
            method="ffill")

    data.fillna(0, inplace=True)
    save_df(data)

    # The timestamp sent by the validator need not be associated with an exact 5m interval
    # It's on the miners to ensure that the time is rounded down to the last completed 5 min candle
    pred_time = round_down_time(datetime.fromisoformat(timestamp))
    matching_row = data[data.index == pred_time]

    # print(pred_time, matching_row)

    # Check if matching_row is empty
    if matching_row.empty:
        print("No matching row found for the given timestamp. Took last one from dataframe.")
        matching_row = data.tail(1)

    input = matching_row.drop(['Adj Close', 'Close'], axis=1)

    prediction = None
    if type == 'lstm':
        input = np.array(input, dtype=np.float32).reshape(1, -1)
        input = np.reshape(input, (1, 1, input.shape[1]))
        prediction = model.predict(input)
        prediction = scaler.inverse_transform(prediction.reshape(1, -1))
        save_model_retraining_args(X_scaled, y_scaled)
    elif type == "arimax":
        prediction = model.predict(n_periods=6, X=data.drop(['Adj Close', 'Close'], axis=1).tail(6).values,
                                   return_conf_int=False)

    return prediction


# Uncomment this section if you wanna do a local test without having to run the miner
# on a subnet. This main block (kinda) mimics the actual validator response being sent
# if (__name__ == '__main__'):
#     #     import yfinance as yf
#     import pickle
#
#     #     import pmdarima as pm
#     #
#     #     # data = prep_data(False)
#     #     # data = yf.download('^GSPC', period='60d', interval='5m')
#     #     # data.reset_index(inplace=True)
#     #     # scaler, X, y = scale_data(data)
#     #     # print(data.tail())
#     #     # # # # mse = create_and_save_base_model_regression(scaler, X, y)
#     #     # # # #
#     #     # # # # model = joblib.load('mining_models/base_linear_regression.joblib')
#     #     # # # #
#     #     # # # ny_timezone = timezone('America/New_York')
#     #     # # # current_time_ny = datetime.now(ny_timezone) + timedelta(days=-1)  # for testing purposes
#     #     # # # timestamp = current_time_ny.isoformat()
#     timestamp = "2024-06-24T15:55:29.139514-04:00"
#     #     # # # #
#     #     # # # from base_miner.model import create_and_save_base_model_lstm
#     #     # # #
#     with open("mining_models/arimax_model.pkl", "rb") as model_f:
#         model = pickle.load(model_f)
#     #     # model = load_model("mining_models/base_lstm_new.h5")
#     #     # model = retrain_and_save(X, y, "mining_models/base_lstm_new.h5")
#     prediction = predict(timestamp, model=model, type='arimax')
#     print(prediction)

# df = pd.read_csv("GSPC.csv").set_index("Datetime")
# print(df.head())
