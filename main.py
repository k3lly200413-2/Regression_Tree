import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
from urllib.request import urlretrieve


def download(file, url):
    if not os.path.isfile(file):
        urlretrieve(url, file)
        
def check_promo2_month(row):
    # Checks if the Month in date is in the Promo2Months which we mapped to numbers 
    # since Date is a single date and not a series we can use .month and we don't need dt.month#
    return row["Date"].month in row["Promo2Months"]


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
    
    print(data.keys())
    
    promo2_started = (
        # if date we are at now is later than the year since the shop started the promo
        (data["Date"].dt.year > data["Promo2SinceYear"])
        | (
            # Or the same year
            (data["Date"].dt.year == data["Promo2SinceYear"])
            # and the week is later than when it started
            & (data["Date"].dt.isocalendar().week >= data["Promo2SinceWeek"])
        )
    )
    
    # Months when promotion was active
    # The form is 
    # 0                     NaN
    # 1         Jan,Apr,Jul,Oct
    # 2         Jan,Apr,Jul,Oct
    # 3                     NaN

    print(data["PromoInterval"])
    
    months_map = {
        np.nan: [],
        "Jan,Apr,Jul,Oct":  [1, 4, 7, 10],
        "Feb,May,Aug,Nov":  [2, 5, 8, 11],
        "Mar,Jun,Sept,Dec": [3, 6, 9, 12]
    }
    
    data["Promo2Months"] = data["PromoInterval"].map(months_map)
    # Now instead of strings we have numbers
    # 0                    []
    # 1         [1, 4, 7, 10]
    # 2         [1, 4, 7, 10]
    
    # Date is of the form 2013-01-01 
    # when we check we take month, e. g. 01
    # we then check if month is inside a list 
    # e. g. if 01 is in [1, 4, 7, 10]
    print(data["Date"])
    
    print(data["Promo2Months"])
    
    is_promo2_month = data.apply(check_promo2_month, axis=1)
    
    # now of the form 
    #
    # 0         False
    # 1          True
    # 2          True
    # 3         False
    # 4         False
    
    print(is_promo2_month)
    
    # We add a column called Promo2Active which checks if the promotion has started 
    # if it has it also has to be in the month that the date is in
    # Basically Checks, has it started? And is it active in this month?
    data["Promo2Active"] = promo2_started & is_promo2_month
    
    print(data["CompetitionOpenSinceMonth"])
    
    data["CompetitionOpen"] = (
        data["CompetitionOpenSinceYear"].isna()
        | (data["Date"].dt.year > data["CompetitionOpenSinceYear"])
        | (
            (data["Date"].dt.year == data["CompetitionOpenSinceYear"])
            & (data["Date"].dt.month >= data["CompetitionOpenSinceMonth"])
        )
    )
    
    # Get the row that has a NaN in the column CompetitionDistance, get the column CompetitionDistance 
    # set this NaN to the max
    data.loc[data["CompetitionDistance"].isna(), "CompetitionDistance"] = data["CopmetitionDistance"].max()
    
    # plt.show()
    
if __name__ == "__main__":
    main()