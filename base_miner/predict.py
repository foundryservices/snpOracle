# Import modules that already exist or can be installed using pip
from datetime import datetime
from pytz import timezone
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from base_miner.model import create_and_save_base_model_regression

# import custom defined files
from base_miner.get_data import prep_data, scale_data, round_down_time
from neurons.utils import save_df


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
            type="arimax") -> np.array:
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
    if type == "arimax":
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data['Datetime'] = data['Datetime'].dt.tz_convert("America/New_York")
        data[['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']] = data[
            ['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']].interpolate(
            method="ffill")

    data.fillna(0, inplace=True)
    save_df(data, "./mining_models/GSPC.csv")

    # The timestamp sent by the validator need not be associated with an exact 5m interval
    # It's on the miners to ensure that the time is rounded down to the last completed 5 min candle
    pred_time = round_down_time(datetime.fromisoformat(timestamp))
    matching_chunk = data[data.index <= pred_time]

    print(f'Prediction time: ${pred_time}, Matching row:${matching_chunk}')

    # Check if matching_row is empty
    if matching_chunk.empty:
        print("No matching chunk found for the given timestamp. Took last one from dataframe.")
        matching_chunk = data

    input = matching_chunk.drop(['Adj Close', 'Close'], axis=1)

    prediction = None
    if type == "arimax":
        prediction = model.predict(n_periods=6, X=input.tail(6).values,
                                   return_conf_int=False)
    return prediction


# Uncomment this section if you wanna do a local test without having to run the miner
# on a subnet. This main block (kinda) mimics the actual validator response being sent
if (__name__ == '__main__'):
    import pickle

    timestamp = "2024-06-26T09:55:29.139514-04:00"
    with open("../mining_models/arimax_model.pkl", "rb") as model_f:
        model = pickle.load(model_f)
    prediction = predict(timestamp, model=model, type='arimax')
    print(f'Predition for ARIMAX ${prediction}')
