# Import necessary modules to use for model creation - can be downloaded using pip
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("HF_ACCESS_TOKEN"):
    print("Cannot find a Huggingface Access Token - unable to upload model to Huggingface.")
token = os.getenv("HF_ACCESS_TOKEN")


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

    # LinearRegression model - all hyperparameters are baseline params - should be changed according to your required
    # architecture.
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
    print(f'Mean Squared Error/MSE for ARIMAX: {mse}')

    return mse
