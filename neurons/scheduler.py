import pickle
import time
from datetime import datetime
from threading import Thread
from utils import load_df, merge_dfs
from base_miner.get_data import prep_data


def main():
    while True:
        t = time.localtime()
        if 13 <= t.tm_hour <= 21 and (
                t.tm_min == 5 or t.tm_min == 35) and t.tm_sec == 0 and datetime.today().weekday() < 5:
            saved_df = load_df()
            new_df = prep_data()
            with open("./mining_models/arimax_model.pkl", "rb") as model_f:
                model = pickle.load(model_f)
            daemon = Thread(target=merge_dfs,
                            args=(saved_df, new_df, model, "arimax"),
                            daemon=True, name='Updating model')
            daemon.start()
            print("Started model updating.")


if __name__ == '__main__':
    main()
