import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from urllib.request import urlretrieve

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def print_eval(X, y, model):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    re = mean_absolute_percentage_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"   Mean squared error: {mse:.5}")
    print(f"       Relative error: {re:.5%}")
    print(f"R-squared coefficient: {r2:.5}")

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
    
    # print(data.keys())
    
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

    # print(data["PromoInterval"])
    
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
    # print(data["Date"])
    
    # print(data["Promo2Months"])
    
    is_promo2_month = data.apply(check_promo2_month, axis=1)
    
    # now of the form 
    #
    # 0         False
    # 1          True
    # 2          True
    # 3         False
    # 4         False
    
    # print(is_promo2_month)
    
    # We add a column called Promo2Active which checks if the promotion has started 
    # if it has it also has to be in the month that the date is in
    # Basically Checks, has it started? And is it active in this month?
    data["Promo2Active"] = promo2_started & is_promo2_month
    
    # print(data["CompetitionOpenSinceMonth"])
    
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
    data.loc[data["CompetitionDistance"].isna(), "CompetitionDistance"] = data["CompetitionDistance"].max()
    
    train_test_split_date = "2015-06-19"
    data_train = data[data["Date"] <= train_test_split_date]
    data_val = data[data["Date"] > train_test_split_date]
    
    y_train = data_train["Sales"]
    y_val = data_val["Sales"]
    
    
    # 1. alcune variabili (es. `CompetitionDistance`) 
    #   esprimono effettivamente una quantità o una grandezza 
    #   (sono di tipo "intervallo" o "ratio")
    # 
    # 2. altre variabili (es. `SchoolHoliday`) sono binarie, 
    #   valgono 1 dove una condizione è verificata e 0 dove non si verifica; 
    #   in questo insieme possiamo includere le variabili di tipo booleano, 
    #   convertite in automatico in binarie ove necessario
    # 
    # 3. altre variabili (es. `StoreType`) esprimono dei "codici", 
    #   i numeri servono solamente da identificatori univoci 
    #   (sono di tipo "nominale")
    
    
    numeric_vars = ["CompetitionDistance"]
    binary_vars = ["Promo", "SchoolHoliday", "Promo2Active", "CompetitionOpen"]
    categorical_vars = ["StateHoliday", "StoreType", "Assortment"]
    
    X_train_num = data_train[numeric_vars + binary_vars]
    X_val_num   = data_val[numeric_vars + binary_vars]
    
    l_r_m = Ridge()
    l_r_m.fit(X_train_num, y_train)
    
    # print(l_r_m.score(X_val_num, y_val))
    
    model = Pipeline([
        ("Scaler", StandardScaler()),
        ("Model", Ridge())
    ])
    
    model.fit(X_train_num, y_train)
    
    # print(model.score(X_val_num, y_val))
    
    # Categorical Columns
    X_sample = data_train[["SchoolHoliday", "StateHoliday"]]
    # print(X_sample.head(5))
    
    # Unique values that these can assume are:
    # [0 1]
    # ['0', 'a', 'b', 'c']
    #   Categories (4, str): ['0', 'a', 'b', 'c']
    # print(X_sample["SchoolHoliday"].unique())
    # print(X_sample["StateHoliday"].unique())
    
    # without sparse_output=False
    # <Compressed Sparse Row sparse matrix of dtype 'float64'
	# with 1608220 stored elements and shape (804110, 6)>
    # Coords	Values
    # (0, 0)	1.0
    # (0, 2)	1.0
    # (1, 0)	1.0
    # (1, 2)	1.0
    # (2, 0)	1.0
    # (2, 2)	1.0
    # (3, 0)	1.0
    # (3, 2)	1.0
    # (4, 0)	1.0
    # (4, 2)	1.0
    # (5, 0)	1.0
    # (5, 2)	1.0
    # (6, 0)	1.0
    # (6, 2)	1.0
    # (7, 0)	1.0
    # (7, 2)	1.0
    # (8, 0)	1.0
    # (8, 2)	1.0
    # (9, 0)	1.0
    # (9, 2)	1.0
    # (10, 0)	1.0
    # (10, 2)	1.0
    # (11, 0)	1.0
    # (11, 2)	1.0
    # (12, 0)	1.0
    # :	:
    
    # With sparse_output=False
    #[[1. 0. 1. 0. 0. 0.]
    # [1. 0. 1. 0. 0. 0.]
    # [1. 0. 1. 0. 0. 0.]
    # ...
    # [0. 1. 0. 1. 0. 0.]
    # [0. 1. 0. 1. 0. 0.]
    # [0. 1. 0. 1. 0. 0.]]

    # By adding drop="first" we drop the first column of each category 
    # we do this because we can deduce the result of the first by looking at the
    # other(s)#
    
    #
    # In questo esempio sono state scartate le variabili `SchoolHoliday_0` e 
    #   `StateHoliday_0`

    # - i casi `SchoolHoliday_0` si riconoscono implicitamente da 
    #   `SchoolHoliday_1 = 0`
    # - i casi `StateHoliday_0` si riconoscono da tutte le colonne `StateHoliday` a 0

    # Il vantaggio di questo accorgimento è che si evita di introdurre 
    #   variabili collineari che possono causare problemi in alcuni modelli 
    #   (es. modelli lineari senza regolarizzazione)

    # Lo svantaggio è che si introduce un'asimmetria nella rappresentazione dei dati 
    #   che può causare bias in altri modelli 
    #   (es. modelli lineari con regolarizzazione)
    
    encoder = OneHotEncoder(sparse_output=False)
    
    # print(encoder.fit_transform(X_sample))
    
    # Takes the data and stores the categories (fit), the applies a function to the data 
    # this is done by transform, in this case it just sets either 1s or 0s to the values
    # of these categories, this transform function can be anything else#
    encoder.fit_transform(X_sample)
    
    # print(encoder.get_feature_names_out())
    
    # print(pd.DataFrame(
    #     encoder.transform(X_sample),
    #     columns=encoder.get_feature_names_out()
    # ).head(5))
    
    X_train_cat = data_train[categorical_vars]
    X_val_cat   = data_val[categorical_vars]
    
    encoder = OneHotEncoder()
    
    model = Ridge()
    model.fit(encoder.fit_transform(X_train_cat), y_train)
    
    # print(model.score(encoder.transform(X_val_cat), y_val))
    
    model = Pipeline([
        ("encode", OneHotEncoder()),
        ("Model", Ridge())
    ])
    
    model.fit(X_train_cat, y_train)
    
    print(model.score(X_val_cat, y_val))
    
    
    
    # plt.show()
    
if __name__ == "__main__":
    main()