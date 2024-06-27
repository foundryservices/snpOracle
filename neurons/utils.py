import os
import pickle
import pandas as pd


def load_df(path: str = "./mining_models/GSPC.csv", type="arimax") -> pd.DataFrame or None:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        if type == "lstm":
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df['Datetime'] = df['Datetime'].dt.tz_convert("America/New_York")
        return df
    else:
        print("There's no GSPC.csv file")
        return None


def save_df(df: pd.DataFrame, path: str = "./mining_models/GSPC.csv") -> bool:
    try:
        df.to_csv(path)
        return True
    except Exception as e:
        print(e)
        return False


def update_arimax_model(model, df_diff):
    y_new = df_diff['Adj Close'].values
    X_new = df_diff.drop(["Adj Close", "Close"], axis=1).values

    for new_ob_y, new_ob_X in zip(y_new, X_new):
        model.update(new_ob_y, X=new_ob_X.reshape(1, -1))
    with open("./mining_models/arimax_model.pkl", "wb") as model_f:
        pickle.dump(model, model_f)
    return model


def merge_dfs(df1: pd.DataFrame, df2: pd.DataFrame, model, type: str = "arimax") -> tuple:
    df_diff = None
    if type == "lstm":
        df_diff = df2[~df2['Datetime'].isin(df1['Datetime'])]
    elif type == "arimax":
        df_diff = df2[~df2.index.isin(df1.index)]
    merge_lst = [df1, df_diff]
    merged_df = pd.concat(merge_lst)
    if type == "arimax":
        model = update_arimax_model(model, df_diff)
    if type == "lstm":
        merged_df.drop(['NextClose1', 'NextClose2', 'NextClose3', 'NextClose4', 'NextClose5', 'NextClose6'], axis=1)
        for i in range(1, 7):
            merged_df[f'NextClose{i}'] = merged_df['Close'].shift(-1 * i)
    save_df(merged_df)
    return merged_df, model
