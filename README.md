# IIITB-ML-Project-Rossmann-Store-Sales

Title - Rossman Store Sales
1)Bharadiya Pavan V.(MT2018023)
2)Devarakonda Deepak S.(MT2018031)

You are provided with historical sales data for Rossmann stores. The task is to forecast the "Sales" column for the test set. Please note that you need to predict only for one day.

1)All the packages required to run the code are given in "requirements.txt" file.
2)All the .py files, .csv files,report, .tex file and pickle files except "randomForest_pickle.pkl" are there in zipped folder.
3)You should download "randomForest_pickle.pkl" which is of size 280MB so we uploaded it in google drive 
which you should get from following link and keep it in same folder.


https://drive.google.com/file/d/1R3aQnOi32DB-NOOzGJiiLoajOfsG8Pwr/view?usp=sharing

right click on the folder and press download option. Unzip it and save pickle file with other pickle files.



main.py file contains the object oriented code for the project in which we are training and testing different models,
creating pickle files of the models and finally predicting the values for test.csv and storing it in final_submission.csv.
The different models used are as follows:

1)LinearRegression
2)DecisionTreeRegressor
3)RandomForestRegressor
4)XGBoostRegressor
5)Stacking of RandomForestRegressor and XGBoostRegressor and applying XGBoostRegressor


For testing purpose we have created test.py where we are loading pickles files created using main.py and checking the 
RMSPE values for each individual model and also you can predict for test.csv as well.

So just run test.py file with pickle files.

For data visualization, we have a separate file called visualization.py. By running this file you will get all the data analysis. 
You can find the report for this project in this folder named as Rossman_Store_Sales_report.pdf.


If you want to build all models again and create new pickle files then run main.py
Else run test.py with existing pickle files of models to get the RMSPE for different models and also to
predict values for test.csv and sotre it in final_submission.csv



