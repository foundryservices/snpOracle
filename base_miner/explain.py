import shap
from get_data import *
from model import *
from predict import *



def explain_lstm(model, X_train, X_test, features):
    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)


if __name__ == '__main__':
    data = prep_data(False)
    scaler, X, y = scale_data(data)
    # mse = create_and_save_base_model_regression(scaler, X, y)

    # model = joblib.load('mining_models/base_linear_regression.joblib')
    #
    ny_timezone = timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()

    features = np.array(['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'CCI', 'Momentum'])

    result = create_base_model_lstm(X, y)

    model = result['model']
    X_train = result['X_train']
    X_test = result['X_test']

    explain_lstm(model, X_train, X_test, features)
