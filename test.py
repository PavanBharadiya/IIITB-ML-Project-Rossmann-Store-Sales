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
from os import path

class Visualize:
    data = pd.DataFrame()
    train=pd.DataFrame()
    store=pd.DataFrame()

    def __init__(self):
        print("Correlation")
        self.read_files()
        self.correlation_matrix()
        self.scatter_plot()

    def read_files(self):
        self.train = pd.read_csv('train.csv', engine='python',parse_dates = [2])
        self.store = pd.read_csv('store.csv', engine='python')
        self.data = self.train

    def correlation_matrix(self):
        data = self.data
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        colormap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.show()

    def scatter_plot(self):
        print("Scatter plot between Customers and Sales for each Day of week")
        data = self.data
        f, ax = plt.subplots(7, sharex=True, sharey=True,facecolor = 'brown')
        for i in range(1, 8):
             plot = data[data['DayOfWeek'] == i]
             ax[i - 1].set_title("Day {0}".format(i))
             ax[i - 1].scatter(plot['Customers'], plot['Sales'], label=i)
        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.show()
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


    def Train_Test_Split(self, train , y,size):
        print("\nSplitting data into train and test")
        X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=size, random_state=123)
        return X_train, X_test, y_train, y_test

    def rmspe(self,y,yhat):
            w = self.ToWeight(y)
            rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
            return rmspe

    def ToWeight(self,y):
        w = np.zeros(y.shape, dtype=float)
        ind = y != 0
        w[ind] = 1./(y[ind]**2)
        return w

    def rmspe_xg(self, yhat, y):
        y = np.expm1(y.get_label())
        yhat = np.expm1(yhat)
        return "rmspe", self.rmspe(y,yhat)


    def testRandomForrestRegressor(self, X_test,y_test=[]):
        with open('randomForest_pickle.pkl', 'rb') as f:
            self.__rforest = pickle.load(f)
        self.y_hat = self.__rforest.predict(X_test)
        if len(y_test):
            print("\nRMSPE for RF on validation:",self.rmspe(y_test,self.y_hat))
        return self.y_hat

    def testDecisionTreeRegressor(self, X_test,y_test):
        with open('dtree_pickle.pkl', 'rb') as f:
            self.__dtree = pickle.load(f)
        self.y_hat = self.__dtree.predict(X_test)
        if len(y_test):
            print("\nRMSPE for DecisionTree on validation:",self.rmspe(y_test,self.y_hat))
        return self.y_hat

    def testLinearRegression(self, X_test,y_test=[]):
        with open('LinearRegressor_pickle.pkl', 'rb') as f:
            self.__linear = pickle.load(f)
        self.y_hat = self.__linear.predict(X_test)
        if len(y_test):
            print("\nRMSPE for LinearRegression on validation:",self.rmspe(y_test,self.y_hat))
        return self.y_hat

    def testXGB(self, X_test, y_test=[]):
        with open('XGBoost_pickle.pkl', 'rb') as f:
            self.__xgboost = pickle.load(f)
        self.y_hat1 = np.expm1(self.__xgboost.predict(X_test))
        if len(y_test):
            print("\nRMSPE for XG Boost on validation:",self.rmspe(y_test,self.y_hat1))
        return self.y_hat1

    def testXGBStack(self, X_test, y_test=[]):
        with open('XGBoost_stack_pickle.pkl', 'rb') as f:
            self.__xgboostStack = pickle.load(f)
        self.y_hat1 = np.expm1(self.__xgboostStack.predict(X_test))
        if len(y_test):
            print("\nRMSPE of stacked model on validaton:",self.rmspe(y_test,self.y_hat1))
        return self.y_hat1


if __name__ ==  '__main__':
    #obj=Visualize()
    train = pd.DataFrame()
    test = pd.DataFrame()
    store = pd.DataFrame()

    models =pd.DataFrame()
    test_models = pd.DataFrame()

    obj = RossmanStore()

    train , test , store = obj.read_files()
    train_clean = obj.data_cleaning(train)

    store_new = obj.add_features(train_clean , store)

    train_final, X = obj.build_features(train_clean , store_new)
    test_final, X1 = obj.build_features(test , store_new)


    X_train, X_test, y_train, y_test = obj.Train_Test_Split(train_final[X], train_final['Sales'],0.4)
    print("\nThe features we are using for Training our model:",X_train.columns)
    #########################MODELS#######################

    #LinearRegression
    print("\nTesting Linear Regression")
    predictions_lr = obj.testLinearRegression(X_test,y_test)

    #DecisionTree
    print("\nTesting DecisionTree Regressor")
    predictions_dt = obj.testDecisionTreeRegressor(X_test,y_test)

    #Random Forest Regressor
    print("\nTesting Random Forest Regressor")
    models['RF'] = obj.testRandomForrestRegressor(X_test,y_test)

    #XGBoost
    print("\nTesting XGBoost Regressor")
    models['XGB'] = obj.testXGB(X_test,y_test)

    #Stacking Features
    test_models['RF'] = obj.testRandomForrestRegressor(test_final[X1])
    test_models['XGB'] = obj.testXGB(test_final[X1])

    X_train1, X_test1, y_train1, y_test1 = obj.Train_Test_Split(models, y_test,0.007)

    print("\nValidating stacking model")
    pred = obj.testXGBStack(X_test1,y_test1)

    print("\npredicting Sales using Stacking on test.csv")
    print("\nStoring the predicted results in final_submission.csv")
    predictions = obj.testXGBStack(test_models)
    predictions = predictions.astype(int)
    df = pd.DataFrame({'Id' : test['Id'], 'Sales': predictions})
    df.loc[test['Open'] == 0, 'Sales'] = 1
    df.to_csv('final_submission.csv', index=False)
