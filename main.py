import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
from urllib.request import urlretrieve


def download(file, url):
    if not os.path.isfile(file):
        urlretrieve(url, file)


def main():
    download(   "rossmann-train.csv.gz", "https://github.com/datascienceunibo/dialab2024/raw/main/Regressione_con_Alberi/rossmann-train.csv.gz")
    download(   "rossmann-stores.csv", "https://github.com/datascienceunibo/dialab2024/raw/main/Preprocessing_con_pandas/rossmann-stores.csv")

    data_sales = pd.read_csv(
        "rossmann-train.csv.gz",
        parse_dates=["Date"],
        dtype={"StateHoliday": "category"},
        compression="gzip",
    )
    
    data_sales.sample(10000, random_state=42).plot.scatter("Customers", "Sales")
    
    data_open = data_sales.loc[data_sales["Open"] == 1].drop(columns=["Open"])
    
    data_stores = pd.read_csv("rossmann-stores.csv")
    
    data = pd.merge(data_open, data_stores, left_on=["Store"], right_on=["Store"])
    
    
    
    plt.show()
    
if __name__ == "__main__":
    main()