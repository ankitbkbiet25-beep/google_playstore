import pandas as pd
import numpy as np

def basic_preprocess(df):

    df.drop_duplicates(inplace=True)
    df = df[df["Category"] != "1.9"]
    df.dropna(subset=["Rating"], inplace=True)

    df["Reviews"] = pd.to_numeric(df["Reviews"], errors='coerce')
    df.dropna(subset=["Reviews"], inplace=True)

    def size_convert(size):
        if size == "Varies with device":
            return np.nan
        elif size.endswith("M"):
            return float(size.replace("M", ""))
        elif size.endswith("k"):
            return float(size.replace("k", "")) / 1000
        else:
            return np.nan

    df["Size"] = df["Size"].apply(size_convert)
    df["Size"].fillna(df["Size"].median(), inplace=True)
    
    df[["Primary_genre", "Secondary_genre"]] = df["Genres"].str.split(";", expand=True).fillna("None")

    df["Installs"] = df["Installs"].str.replace(",", "").str.replace("+", "", regex=False)
    df["Installs"] = pd.to_numeric(df["Installs"], errors='coerce')
    df.dropna(subset=["Installs"], inplace=True)

    df["Price"] = df["Price"].str.replace("$", "", regex=False)
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
    df["Price"].fillna(0, inplace=True)

    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors='coerce')
    df['Update_Year']=df["Last Updated"].dt.year.astype('int64')
    df["update_month"]=df["Last Updated"].dt.month.astype('int64')
    df["Last Updated"]=df["Last Updated"].dt.date


    return df
