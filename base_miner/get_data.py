# Import required models
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ta
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import yfinance as yf


def prep_data(drop_na: bool = True, type: str = "arimax") -> DataFrame:
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
    data = yf.download('^GSPC', period='60d', interval='5m')

    # Calculate technical indicators - all technical indicators computed here are based on the 5m data
    # For example - SMA_50, is not a 50-day moving average, but is instead a 50 5m moving average
    # since the granularity of the data we procure is at a 5m interval. 
    # TREND

    # SMA_50
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    # Exponential Moving Average (EMA) as an imput for NaN values
    mean_SMA_50 = ta.trend.ema_indicator(data['Close'], window=2, fillna=True)
    # Replace NaNs in column SMA_50 with the
    # Exponential Moving Average (EMA) window=2 of values in the same column
    data['SMA_50'].fillna(value=mean_SMA_50, inplace=True)

    # SMA_200
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    # Exponential Moving Average (EMA) as an imput for NaN values
    mean_SMA_200 = ta.trend.ema_indicator(data['Close'], window=8, fillna=True)
    # Replace NaNs in column SMA_200 with the
    # Exponential Moving Average (EMA) window=2x4 of values in the same column
    data['SMA_200'].fillna(value=mean_SMA_200, inplace=True)

    # CCI
    # Commodity Channel Index (CCI)
    # CCI measures the difference between a security’s price change and its average price change.
    # High positive readings indicate that prices are well above their average, which is a show of strength.
    # Low negative readings indicate that prices are well below their average, which is a show of weakness.
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=20, constant=0.015,
                                        fillna=True).cci()

    # Exponential Moving Average (EMA) as an imput for NaN values
    mean_CCI = data['CCI'].fillna(0)
    # mean_CCI = data['CCI'].rolling(window=3).mean()
    # Replace NaNs in column CCI with the
    # Exponential Moving Average (EMA) window=2x4 of values in the same column
    data['CCI'].fillna(value=mean_CCI, inplace=True)

    # VOLATILITY

    # Initialize Bollinger Bands Indicator
    # indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)

    # Add Bollinger Bands features
    data['bb_bbm'] = ta.volatility.bollinger_mavg(close=data["Close"], window=50)
    # Exponential Moving Average (EMA) as an imput for NaN values
    mean_bb_bbm = ta.trend.ema_indicator(data['Close'], window=10, fillna=True)
    # Replace NaNs in column bb_bbm with the
    # Exponential Moving Average (EMA) window=10 of values in the same column
    data['bb_bbm'].fillna(value=mean_bb_bbm, inplace=True)

    # bb_bbh
    data['bb_bbh'] = ta.volatility.bollinger_hband(close=data["Close"], window=20, window_dev=2, fillna=True)
    # Exponential Moving Average (EMA) as an imput for NaN values
    mean_bb_bbh = ta.trend.ema_indicator(data['Close'], window=5, fillna=True)
    # Replace NaNs in column bb_bbm with the
    # Exponential Moving Average (EMA) window=5 of values in the same column
    data['bb_bbh'].fillna(value=mean_bb_bbh, inplace=True)

    # bb_bbl
    data['bb_bbl'] = ta.volatility.bollinger_lband(data['Close'], window=20, window_dev=2, fillna=True)
    # Exponential Moving Average (EMA) as an imput for NaN values
    mean_bb_bbl = ta.trend.ema_indicator(data['Close'], window=5, fillna=True)
    # Replace NaNs in column bb_bbm with the
    # Exponential Moving Average (EMA) window=5 of values in the same column
    data['bb_bbl'].fillna(value=mean_bb_bbl, inplace=True)

    # MOMENTUM

    # Relative Strength Index (RSI)
    # Compares the magnitude of recent gains and losses over a specified time period to measure speed and change of price movements of a security.
    # It is primarily used to attempt to identify overbought or oversold conditions in the trading of an asset.
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14, fillna=True).rsi()
    mean_RSI = data['Momentum'].fillna(0)
    data['RSI'].fillna(value=mean_RSI, inplace=True)


    data['Momentum'] = ta.momentum.ROCIndicator(data['Close'], window=14, fillna=True).roc()
    mean_Momentum = data['Momentum'].fillna(0)
    data['Momentum'].fillna(value=mean_Momentum, inplace=True)

    # VOLUME
    data['LastIntervalReturn'] = (data['Close'].shift(0) / data['Close'].shift(-1)) - 1
    mean_LastIntervalReturn = data['LastIntervalReturn'].fillna(0)
    data['LastIntervalReturn'].fillna(value=mean_LastIntervalReturn, inplace=True)

    data['VolumeChange'] = (data['Volume'].shift(0) / data['Volume'].shift(-1)) - 1
    mean_VolumeChange = data['VolumeChange'].fillna(0)
    data['VolumeChange'].fillna(value=mean_VolumeChange, inplace=True)

    if type == "arimax":
        for i in range(1, 7):
            data[f'NextClose{i}'] = data['Close'].shift(-1 * i)

    # Drop NaN values
    if (drop_na):
        data.dropna(inplace=True)

    # data.reset_index(inplace=True)

    return data


def round_down_time(dt: datetime, interval_minutes: int = 5) -> str:
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
    rounded_dt = dt - timedelta(minutes=dt.minute % interval_minutes,
                                seconds=dt.second,
                                microseconds=dt.microsecond)

    return rounded_dt.isoformat()  #


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
    X = data.drop(["Adj Close", "Close"], axis=1).values

    # Prepare target variable
    y = data[['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6']].values

    y = y.reshape(-1, 6)

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    return scaler, X_scaled, y_scaled
