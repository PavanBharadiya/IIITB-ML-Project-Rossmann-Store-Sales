import pandas as pd
import numpy as np
import string
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path
import pickle
import os.path
import datetime as dt
from os import path

class Visualize:

    def __init__(self):
        print("\nStarted EDA")
        self.train1 = pd.DataFrame()

    def data_cleaning(self , train):
        train['Year'] = train.Date.dt.year
        train['Month'] = train.Date.dt.month
        train['DayOfMonth'] = train.Date.dt.day
        train['WeekOfYear'] = train.Date.dt.weekofyear
        train['DayOfYear'] = train.Date.dt.dayofyear
        train['DayOfWeek'] = train.Date.dt.dayofweek
        train["StateHolidayBinary"] = train["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
        self.train1 = train.copy()
        train = train.loc[train.Open == 1]
        train = train.loc[train['Sales'] > 0]
        return train


    def data_analysis(self,data):
        f, ax = plt.subplots(figsize = (10, 10))
        corr = self.train1[data.columns].corr()
        mask = np.zeros_like(corr, dtype = np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask = mask,annot= True,square = True,fmt = '0.2f', linewidths = .5, ax = ax)
        plt.title("Correlation Heatmap", fontsize=20)
        plt.show()

        fig, (axis1) = plt.subplots(1, 1, figsize=(6, 6))
        data["Sales"].plot(kind="hist", bins=70, xlim=(0, 20000), ax=axis1)
        fig.tight_layout()
        plt.xlabel('Sales')
        plt.show()
        print("Plotted Frequency of Sales Values for open stores")

        fig, (axis1) = plt.subplots(1, 1, figsize=(6, 6))
        data["Customers"].plot(kind="hist", bins=70, xlim=(0, 4000), ax=axis1)
        fig.tight_layout()
        plt.xlabel('Customers')
        plt.show()
        print("Plotted Frequency of Customers Values (for Open Stores)")

        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(6, 6))
        sns.barplot(x="DayOfWeek", y="Sales", data=data, order=[1, 2, 3, 4, 5, 6, 7], ax=axis1, ci=None)
        sns.barplot(x="DayOfWeek", y="Customers", data=data, order=[1, 2, 3, 4, 5, 6, 7], ax=axis2, ci=None)
        fig.tight_layout()
        plt.show()
        print("Plotted Avg. Sales & Customers (by Day of Week)")

        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(6, 6))
        sns.barplot(x="Month", y="Sales", data=data, ax=axis1, ci=None)
        sns.barplot(x="Month", y="Customers", data=data, ax=axis2, ci=None)
        fig.tight_layout()
        plt.show()
        print("Plotted Avg. Sales & Customers (by Month)")

        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(6, 6))
        sns.barplot(x="Promo", y="Sales", data=data, ax=axis1, ci=None)
        sns.barplot(x="Promo", y="Customers", data=data, ax=axis2, ci=None)
        fig.tight_layout()
        plt.show()
        print("Plotted Avg. Sales & Customers (by Promo)")

        fig, (axis1) = plt.subplots(1, 1, figsize=(5, 5))
        sns.countplot(x="StateHoliday", data=self.train1)
        fig.tight_layout()
        plt.show()
        print("Plotted No. of State Holidays")

        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(6, 6))
        sns.barplot(x="StateHolidayBinary", y="Sales", data=data, ax=axis1, ci=None)
        sns.barplot(x="StateHolidayBinary", y="Customers", data=data, ax=axis2, ci=None)
        fig.tight_layout()
        plt.show()
        print("Plotted Avg. Sales & Customers (by State Holiday Binary for Open Stores)")


    def data_analysis_after_FE(self,data):
        fig, (axis1) = plt.subplots(1, 1, figsize=(7, 7))
        data.plot(kind="scatter", x="CompetitionDistance", y="AvgSales", ax=axis1)
        fig.tight_layout()
        plt.show()
        print("Plotted Competition Distance vs. Avg. Sales")

        # Generate plot for CompetitionDistance vs. Avg. Customers
        fig, (axis1) = plt.subplots(1, 1, figsize=(7, 7))
        data.plot(kind="scatter", x="CompetitionDistance", y="AvgCustomers", ax=axis1)
        fig.tight_layout()
        plt.show()
        print("Plotted Competition Distance vs. Avg. Customers")


        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(7, 7))
        sns.barplot(x="StoreType", y="AvgSales", data=data, order=[0,1,2,3], ax=axis1, ci=None)
        sns.barplot(x="StoreType", y="AvgCustomers", data=data, order=[0,1,2,3], ax=axis2, ci=None)
        fig.tight_layout()
        plt.show()
        print("Plotted Avg. Sales & Customers (by Store Type)")
############################################################################################################################################
class RossmanStore:
    train = pd.DataFrame()
    test = pd.DataFrame()
    store = pd.DataFrame()

    #Constructor
    def __init__(self):
        print("Initialize random forest, XG boost and XG boost for stacking models")
        self.__linear = LinearRegression()
        self.__dtree = DecisionTreeRegressor(max_depth = 25,min_samples_leaf = 50,random_state = 10)
        self.__rforest = RandomForestRegressor(n_estimators = 100,max_depth = 100,min_samples_leaf = 1, min_samples_split = 20,bootstrap = True, verbose = 0,n_jobs = 1)
        self.__xgboost = xgb.XGBRegressor(n_jobs = -1,n_estimators = 4000,learning_rate = 0.1,max_depth = 2,min_child_weight = 2,subsample = 0.8,colsample_bytree = 0.8,tree_method = 'exact',reg_alpha = 0.05,silent = 1,random_state = 1023)
        self.__xgboostStack = xgb.XGBRegressor(n_jobs = -1,n_estimators = 4000,learning_rate = 0.1,max_depth = 2,min_child_weight = 2,subsample = 0.8,colsample_bytree = 0.8,tree_method = 'exact',reg_alpha = 0.05,silent = 1,random_state = 1023)
        #self.read_files()

    def read_files(self):
        print("\nReading the data")
        types = {'StateHoliday': np.dtype(str)}
        self.train = pd.read_csv("train.csv", parse_dates=[2], dtype=types)
        self.test = pd.read_csv("test.csv", parse_dates=[3], dtype=types)
        self.store = pd.read_csv("store.csv")
        print("\nTrain, Store, Test data files are read and returned to main function")
        return self.train , self.test , self.store

    def data_cleaning(self , train):
        print("\nBefore data cleaning",train.shape)
        train = train.loc[train.Open == 1]
        train = train.loc[train['Sales'] > 0]
        print("\nAfter removing all entries where Store is closed and Sales is 0", train.shape)
        train = train.loc[train['DayOfWeek'].isin([2,3,4,5])]
        #train = train.loc[train['DayOfWeek'] == 7]
        return train

    def add_features(self , train ,store):
        print("\nAdding features for Store:")
        train['SalesPerCustomer'] = train['Sales'] / train['Customers']
        avg_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].mean()
        avg_store.rename(columns=lambda x: 'Avg' + x, inplace=True)
        store = pd.merge(avg_store.reset_index(), store, on='Store')
        med_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].median()
        med_store.rename(columns=lambda x: 'Med' + x, inplace=True)
        store = pd.merge(med_store.reset_index(), store, on='Store')

        return store

    def build_features(self , train , store):
        print("\nStarted feature Engineering")
        # Convert string types into integers
        store['StoreType'] = store['StoreType'].astype('category').cat.codes
        store['Assortment'] = store['Assortment'].astype('category').cat.codes
        train["StateHoliday"] = train["StateHoliday"].astype('category').cat.codes
        data_store = pd.merge(train, store, on='Store', how='left')
        # remove NaNs
        NaN_replace = 0
        data_store.fillna(NaN_replace, inplace=True)
        data_store['Year'] = data_store.Date.dt.year
        data_store['Month'] = data_store.Date.dt.month
        data_store['DayOfMonth'] = data_store.Date.dt.day
        data_store['WeekOfYear'] = data_store.Date.dt.weekofyear
        data_store['DayOfYear'] = data_store.Date.dt.dayofyear
        data_store['DayOfWeek'] = data_store.Date.dt.dayofweek
        # Number of months that competition has existed for
        data_store['MonthsCompetitionOpen'] = 12 * (data_store['Year'] - data_store['CompetitionOpenSinceYear']) + (data_store['Month'] - data_store['CompetitionOpenSinceMonth'])
        data_store.loc[data_store['CompetitionOpenSinceYear'] ==NaN_replace, 'MonthsCompetitionOpen'] = NaN_replace

         # Number of weeks that promotion has existed for
        data_store['WeeksPromoOpen'] = 12 * (data_store['Year'] - data_store['Promo2SinceYear']) + (data_store['Date'].dt.weekofyear - data_store['Promo2SinceWeek']) / 4.0
        data_store.loc[data_store['Promo2SinceYear'] == NaN_replace, 'WeeksPromoOpen'] = NaN_replace

        toInt = ['CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek', 'Promo2SinceYear', 'MonthsCompetitionOpen', 'WeeksPromoOpen'    ]
        data_store[toInt] = data_store[toInt].astype(int)
        data_store['RootOfCustomers'] = np.sqrt(data_store['Customers'])
        data_store['LogOfCustomers'] = np.log1p(data_store['Customers'])
        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        data_store['monthStr'] = data_store.Month.map(month2str)
        data_store.loc[data_store.PromoInterval == 0, 'PromoInterval'] = ''
        data_store['IsPromoMonth'] = 0
        for interval in data_store.PromoInterval.unique():
            if interval != '':
                for month in interval.split(','):
                    data_store.loc[(data_store.monthStr == month) & (data_store.PromoInterval == interval), 'IsPromoMonth'] = 1
        X = ['Store', 'Customers','CompetitionDistance', 'Promo', 'Promo2','StateHoliday','StoreType','Assortment','AvgSales','AvgCustomers','AvgSalesPerCustomer','MedSales','MedCustomers', 'MedSalesPerCustomer','DayOfWeek','WeekOfYear','DayOfMonth','Month','Year','DayOfYear','IsPromoMonth','MonthsCompetitionOpen','WeeksPromoOpen']
        print("\nFinished feature Engineering")
        return data_store, X






if __name__ == "__main__":
    train = pd.DataFrame()
    test = pd.DataFrame()
    store = pd.DataFrame()

    obj = RossmanStore()
    visualize = Visualize()
    train, test, store = obj.read_files()

    #data visualization before feature Engineering
    train_vis = visualize.data_cleaning(train)
    visualize.data_analysis(train_vis)

    #data visualization after feature Engineering
    train_clean = obj.data_cleaning(train)
    store_new = obj.add_features(train_clean , store)
    train_final, X = obj.build_features(train_clean , store_new)

    visualize.data_analysis_after_FE(train_final)
