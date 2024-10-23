# developer: Foundry Digital
# Copyright © 2023 Foundry Digital

# Import required models
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import ta
import yfinance as yf
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def prep_data(drop_na: bool = True) -> DataFrame:
    """
    Prepare data by calling Yahoo Finance SDK & computing technical indicators.

    The function gets the last 60 day data for the S&P 500 index at a 5m granularity
    and then computes the necessary technical indicators, resets index and drops rows
    with NA values if mentioned.

    Input:
        :param drop_na: The drop_na flag is used to tell the function whether to drop rows
                        with nan values or keep them.
        :type drop_na: bool

    Output:
        :returns: A pandas dataframe with the OHLCV data, along with the some technical indicators.
                  The dataframe also has the next close as a column to predict future close price using
                  current data.
        :rtype: pd.DataFrame
    """
    # Fetch S&P 500 data - when capturing data any interval, the max we can go back is 60 days
    # using Yahoo Finance's Python SDK
    data = yf.download("^GSPC", period="60d", interval="5m")

    # Calculate technical indicators - all technical indicators computed here are based on the 5m data
    # For example - SMA_50, is not a 50-day moving average, but is instead a 50 5m moving average
    # since the granularity of the data we procure is at a 5m interval.
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    data["CCI"] = ta.trend.CCIIndicator(
        data["High"], data["Low"], data["Close"]
    ).cci()
    data["Momentum"] = ta.momentum.ROCIndicator(data["Close"]).roc()
    for i in range(1, 7):
        data[f"NextClose{i}"] = data["Close"].shift(-1 * i)

    # Drop NaN values
    if drop_na:
        data.dropna(inplace=True)

    data.reset_index(inplace=True)

    return data


def round_down_time(dt: datetime, interval_minutes: int = 5) -> datetime:
    """
    Find the time of the last started `interval_minutes`-min interval, given a datetime

    Input:
        :param dt: The datetime value which needs to be rounded down to the last 5m interval
        :type dt: datetime

        :param interval_minutes: interval_minutes gives the interval we want to round down by and
                            the default is set to 5 since the price predictions being done
                            now are on a 5m interval
        :type interval_minutes: int

    Output:
        :returns: A datetime of the last started 5m interval
        :rtype: datetime
    """

    # Round down the time to the nearest interval
    rounded_dt = dt - timedelta(
        minutes=dt.minute % interval_minutes,
        seconds=dt.second,
        microseconds=dt.microsecond,
    )

    return rounded_dt


def scale_data(data: DataFrame) -> Tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    """
    Normalize the data procured from yahoo finance between 0 & 1

    Function takes a dataframe as an input, scales the input and output features and
    then returns the scaler itself, along with the scaled inputs and outputs. Scaler is
    returned to ensure that the output being predicted can be rescaled back to a proper
    value.

    Input:
        :param data: The S&P 500 data procured from a certain source at a 5m granularity
        :type data: pd.DataFrame

    Output:
        :returns: A tuple of 3 values -
                - scaler : which is the scaler used to scale the data (MixMaxScaler)
                - X_scaled : input/features to the model scaled (np.ndarray)
                - y_scaled : target variable of the model scaled (np.ndarray)
    """
    X = data[
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
    ].values

    # Prepare target variable
    y = data[
        [
            "NextClose1",
            "NextClose2",
            "NextClose3",
            "NextClose4",
            "NextClose5",
            "NextClose6",
        ]
    ].values

    y = y.reshape(-1, 6)

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    return scaler, X_scaled, y_scaled
