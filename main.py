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


    def plot_importance(self,model,X):
        print("\nPlotting Importance")
        k = list(zip(X, model.feature_importances_))
        k.sort(key=lambda tup: tup[1])
        labels, vals = zip(*k)
        plt.barh(np.arange(len(X)), vals, align='center')
        plt.yticks(np.arange(len(X)), labels)
        plt.show()

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

    def trainRandomForrestRegressor(self,X_train,y_train,X):
        self.__rforest.fit(X_train, y_train)
        self.plot_importance(self.__rforest,X)
        with open('randomForest_pickle.pkl', 'wb') as f:
            pickle.dump(self.__rforest,f)


    def testRandomForrestRegressor(self, X_test,y_test=[]):
        self.y_hat = self.__rforest.predict(X_test)
        if len(y_test):
            print("\nRMSPE for RF on validation:",self.rmspe(y_test,self.y_hat))
        return self.y_hat

    def trainDecisionTreeRegressor(self,X_train,y_train):
        self.__dtree.fit(X_train, y_train)
        with open('dtree_pickle.pkl', 'wb') as f:
            pickle.dump(self.__dtree,f)

    def testDecisionTreeRegressor(self, X_test,y_test):
        self.y_hat = self.__dtree.predict(X_test)
        if len(y_test):
            print("\nRMSPE for DecisionTree on validation:",self.rmspe(y_test,self.y_hat))
        return self.y_hat

    def trainLinearRegresson(self,X_train,y_train):
        self.__linear.fit(X_train, y_train)
        with open('LinearRegressor_pickle.pkl', 'wb') as f:
            pickle.dump(self.__linear,f)

    def testLinearRegression(self, X_test,y_test=[]):
        self.y_hat = self.__linear.predict(X_test)
        if len(y_test):
            print("\nRMSPE for LinearRegression on validation:",self.rmspe(y_test,self.y_hat))
        return self.y_hat

    def trainXGB(self, X_train, y_train):
        self.__xgboost.fit(X_train, np.log1p(y_train), eval_set = [(X_train, np.log1p(y_train))], eval_metric = self.rmspe_xg, early_stopping_rounds = 100,verbose = False)
        xgb.plot_importance(self.__xgboost)
        plt.show()
        with open('XGBoost_pickle.pkl', 'wb') as f:
            pickle.dump(self.__xgboost, f)

    def testXGB(self, X_test, y_test=[]):
        self.y_hat1 = np.expm1(self.__xgboost.predict(X_test))
        if len(y_test):
            print("\nRMSPE for XG Boost on validation:",self.rmspe(y_test,self.y_hat1))
        return self.y_hat1

    def trainXGBStack(self, X_train, y_train):
        self.__xgboostStack.fit(X_train, np.log1p(y_train), eval_set = [(X_train, np.log1p(y_train))], eval_metric = self.rmspe_xg, early_stopping_rounds = 100, verbose = False)
        xgb.plot_importance(self.__xgboostStack)
        plt.show()
        with open('XGBoost_stack_pickle.pkl', 'wb') as f:
            pickle.dump(self.__xgboostStack, f)

    def testXGBStack(self, X_test, y_test=[]):
        self.y_hat1 = np.expm1(self.__xgboostStack.predict(X_test))
        if len(y_test):
            print("\nRMSPE of stacked model on validaton:",self.rmspe(y_test,self.y_hat1))
        return self.y_hat1


if __name__ == "__main__":
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
    print("\nTraining Linear Regression")
    obj.trainLinearRegresson(X_train,y_train)
    print("\nvalidating Linear Regression")
    predictions_lr = obj.testLinearRegression(X_test,y_test)


    #DecisionTree
    print("\nTraining DecisionTree")
    obj.trainDecisionTreeRegressor(X_train,y_train)
    print("\nprint("\npredicting Sales using Stacking on test.csv")alidating DecisionTree Regressor")
    predictions_dt = obj.testDecisionTreeRegressor(X_test,y_test)

    #Random Forest Regressor
    print("\nTraining Random Forest Regressor")
    obj.trainRandomForrestRegressor(X_train,y_train,X)
    print("\nValidating Random Forest Regressor")
    models['RF'] = obj.testRandomForrestRegressor(X_test,y_test)


    #XGBoost
    print("\nTraining XGBoost Regressor")
    obj.trainXGB(X_train,y_train)
    print("\nValidating XGBoost Regressor")
    models['XGB'] = obj.testXGB(X_test,y_test)

    #Stacking Features
    test_models['RF'] = obj.testRandomForrestRegressor(test_final[X1])
    test_models['XGB'] = obj.testXGB(test_final[X1])


    X_train1, X_test1, y_train1, y_test1 = obj.Train_Test_Split(models, y_test,0.007)

    print("\nTraining XGBoost model on RF and XGB(Stacking)")
    obj.trainXGBStack(X_train1,y_train1)

    print("\npredicting Sales using Stacking on test.csv")
    predictions = obj.testXGBStack(test_models,[])
    predictions = predictions.astype(int)
    df = pd.DataFrame({'Id' : test['Id'], 'Sales': predictions})
    df.loc[test['Open'] == 0, 'Sales'] = 1
    df.to_csv('final_submission.csv', index=False)
