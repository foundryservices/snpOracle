import json
import time
import numpy as np
from datetime import datetime
from threading import Thread
from model import retrain_and_save


def load_scaled_args():
    with open("scaled_vars.json", "r") as file:
        scaled_vars = json.load(file)

    return np.array(scaled_vars["X_scaled"]), np.array(scaled_vars["y_scaled"])


def main():
    while True:
        t = time.localtime()
        if 13 <= t.tm_hour <= 21 and (
                t.tm_min == 40 or t.tm_min == 10) and t.tm_sec == 0 and datetime.today().weekday() < 5:
            X_scaled, y_scaled = load_scaled_args()
            daemon = Thread(target=retrain_and_save,
                            args=(X_scaled, y_scaled, "base_miner/mining_models/base_lstm_new.h5"),
                            daemon=True, name='Retraining model')
            daemon.start()
            print("Started model retraining.")


if __name__ == '__main__':
    main()
