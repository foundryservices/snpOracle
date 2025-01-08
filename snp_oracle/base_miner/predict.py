from datetime import datetime

import numpy as np
import pandas as pd
from pytz import timezone
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from snp_oracle.base_miner.get_data import prep_data, round_down_time, scale_data
from snp_oracle.base_miner.model import create_and_save_base_model_lstm


def predict(timestamp: datetime, scaler: MinMaxScaler, model, type: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Predicts the close price of the next 5m interval and returns the input data used for prediction.

    The predict function also ensures that the data is procured - using yahoo finance's python module,
    prepares the data and gets basic technical analysis metrics, and finally predicts the model
    and scales it based on the scaler used previously to create the model.

    Input:
        :param timestamp: The timestamp of the instant at which the request is sent by the validator
        :type timestamp: datetime.datetime

        :param scaler: The scaler used to scale the inputs during model training process
        :type scaler: sklearn.preprocessing.MinMaxScaler

        :param model: The model used to make the predictions - in this case a .h5 file
        :type model: tensorflow.keras.Model

        :param type: The type of model being used (e.g. "regression" or "lstm")
        :type type: str

    Output:
        :returns: A tuple containing (prediction array, input data used for prediction)
        :rtype: tuple[numpy.ndarray, pandas.DataFrame]
    """
    # calling this to get the data - the information passed by the validator contains
    # only a timestamp, it is on the miners to get the data and prepare is according to their requirements
    data = prep_data(drop_na=False)

    # Ensuring that the Datetime column in the data procured from yahoo finance is truly a datetime object
    data["Datetime"] = pd.to_datetime(data["Datetime"])

    # The timestamp sent by the validator need not be associated with an exact 5m interval
    # It's on the miners to ensure that the time is rounded down to the last completed 5 min candle
    pred_time = round_down_time(datetime.fromisoformat(timestamp))

    matching_row = data[data["Datetime"] == pred_time]

    print(pred_time, matching_row)

    # Check if matching_row is empty
    if matching_row.empty:
        print("No matching row found for the given timestamp.")
        return 0.0, pd.DataFrame()

    # data.to_csv('mining_models/base_miner_data.csv')
    input = matching_row[
        [
            "Open",
            "High",
            "Low",
            "Volume",
            "SMA_50",
            "SMA_200",
            "RSI",
            "CCI",
            "Momentum",
        ]
    ]

    input_df = input.copy()

    if type != "regression":
        input = np.array(input, dtype=np.float32).reshape(1, -1)
        input = np.reshape(input, (1, 1, input.shape[1]))
        print(input)

    prediction = model.predict(input)
    if type != "regression":
        prediction = scaler.inverse_transform(prediction.reshape(1, -1))

    print(prediction)
    return prediction, input_df


# Uncomment this section if you wanna do a local test without having to run the miner
# on a subnet. This main block (kinda) mimics the actual validator response being sent
if __name__ == "__main__":
    data = prep_data()
    scaler, X, y = scale_data(data)
    # mse = create_and_save_base_model_regression(scaler, X, y)

    # model = joblib.load('mining_models/base_linear_regression.joblib')
    #
    ny_timezone = timezone("America/New_York")
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()

    mse = create_and_save_base_model_lstm(scaler, X, y)
    model = load_model("mining_models/base_lstm_new.h5")
    prediction, _ = predict(timestamp, scaler, model, type="lstm")
    print(prediction[0])
