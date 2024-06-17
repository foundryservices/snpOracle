# developer: Foundry Digital
# Copyright © 2023 Foundry Digital

# Import necessary modules to use for model creation - can be downloaded using pip
import joblib
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# tensorflow.compat.v1.disable_v2_behavior()

load_dotenv()

if not os.getenv("HF_ACCESS_TOKEN"):
    print("Cannot find a Huggingface Access Token - unable to upload model to Huggingface.")
token = os.getenv("HF_ACCESS_TOKEN")


def retrain_and_save(X_scaled: np.ndarray, y_scaled: np.ndarray):
    """
    For testing purposes only!
    Args:
        scaler:
        X_scaled:
        y_scaled:

    Returns:
        Model object
    """
    model_name = "mining_models/base_lstm_new"

    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # LSTM model - all hyperparameters are baseline params - should be changed according to your required
    # architecture. LSTMs are also not the only way to do this, can be done using any algo deemed fit by
    # the creators of the miner.
    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=6))

    # Compile the model
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])

    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=3,
                                                              mode='min')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping])
    try:
        model.save(f'{model_name}.h5')
        print(f"Successfully saved model in: {model_name}.h5")
    except Exception as e:
        print(e)

    return model
    # return {
    #     "model": model,
    #     "X_train": X_train,
    #     "X_test": X_test,
    #     "y_train": y_train,
    #     "y_test": y_test
    # }


def create_and_save_base_model_lstm(scaler: MinMaxScaler, X_scaled: np.ndarray, y_scaled: np.ndarray) -> float:
    """
    Base model that can be created for predicting the S&P 500 close price

    The function creates a base model, given a scaler, inputs and outputs, and
    stores the model weights as a .h5 file in the mining_models/ folder. The model
    architecture and model name given now is a placeholder, can (and should)
    be changed by miners to build more robust models.

    Input:
        :param scaler: The scaler used to scale the inputs during model training process
        :type scaler: sklearn.preprocessing.MinMaxScaler

        :param X_scaled: The already scaled input data that will be used by the model to train and test
        :type X_scaled: np.ndarray

        :param y_scaled: The already scaled output data that will be used by the model to train and test
        :type y_scaled: np.ndarray
    
    Output:
        :returns: The MSE of the model on the test data
        :rtype: float
    """
    model_name = "mining_models/base_lstm_new"

    # Reshape input for LSTM
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # LSTM model - all hyperparameters are baseline params - should be changed according to your required
    # architecture. LSTMs are also not the only way to do this, can be done using any algo deemed fit by
    # the creators of the miner.
    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=6))

    # Compile the model
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])

    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=3,
                                                              mode='min')

    # Train the model
    model.fit(X_train, y_train, epochs=500, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping])
    model.save(f'{model_name}.h5')

    # api = HfApi()
    # api.upload_file(
    #     path_or_fileobj="mining_models/base_lstm_new.h5",
    #     path_in_repo=f"{model_name}.h5",
    #     repo_id="foundryservices/bittensor-sn28-base-lstm",
    #     repo_type="model",
    #     token=token
    # )

    # Predict the prices - this is just for a local test, this prediction just allows
    # miners to assess the performance of their models on real data.
    predicted_prices = model.predict(X_test)
    #
    # # Rescale back to original range
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 6))
    #
    # # Evaluate
    mse = mean_squared_error(y_test_rescaled, predicted_prices)
    print(f'Mean Squared Error: {mse}')
    #
    return mse


def create_and_save_base_model_regression(scaler: MinMaxScaler, X_scaled: np.ndarray, y_scaled: np.ndarray) -> float:
    """
    Base model that can be created for predicting the S&P 500 close price

    The function creates a base model, given a scaler, inputs and outputs, and
    stores the model weights as a .h5 file in the mining_models/ folder. The model
    architecture and model name given now is a placeholder, can (and should)
    be changed by miners to build more robust models.

    Input:
        :param scaler: The scaler used to scale the inputs during model training process
        :type scaler: sklearn.preprocessing.MinMaxScaler

        :param X_scaled: The already scaled input data that will be used by the model to train and test
        :type X_scaled: np.ndarray

        :param y_scaled: The already scaled output data that will be used by the model to train and test
        :type y_scaled: np.ndarray
    
    Output:
        :returns: The MSE of the model on the test data
        :rtype: float
    """
    model_name = "mining_models/base_linear_regression"

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # LSTM model - all hyperparameters are baseline params - should be changed according to your required
    # architecture. LSTMs are also not the only way to do this, can be done using any algo deemed fit by
    # the creators of the miner.
    model = LinearRegression()
    model.fit(X_train, y_train)

    '''with h5py.File(f'{model_name}.h5', 'w') as hf:
        hf.create_dataset('coefficients', data=model.coef_)
        hf.create_dataset('intercept', data=model.intercept_)'''
    joblib.dump(model, f"{model_name}.joblib")

    # Predict the prices - this is just for a local test, this prediction just allows
    # miners to assess the performance of their models on real data.
    predicted_prices = model.predict(X_test)

    # Rescale back to original range
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate
    mse = mean_squared_error(y_test_rescaled, predicted_prices)
    print(f'Mean Squared Error: {mse}')

    return mse
