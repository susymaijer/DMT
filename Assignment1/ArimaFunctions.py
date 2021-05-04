#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.metrics import mean_squared_error, r2_score


# ### Arima functions

# In[4]:


perc_train = 0.8
sns.set_style("darkgrid")

def getPersonMoodArimaSet(df, person):
    """
        Get the arima mood dataset for a given person.
        
        @param string person:       the id of the person
    """
    # Get mood data for a particular person and set time as index
    mood = df.loc[df["id"] == person][["mood", "time"]]
    mood.set_index(pd.DatetimeIndex(pd.to_datetime(mood.time)), inplace = True)
    mood.drop('time',axis=1,inplace=True)

    # Arima needs a full date range, so fill missing dates
    full_time_range = pd.date_range(mood.index.min(), mood.index.max())
    mood.reindex(full_time_range)
    mood = mood.asfreq('d') # necessary for arima
    
    return mood

def is_correct(x, y):
    """ Returns true is x and y are within 0.5 of eachother """
    return abs(float(x) - float(y)) < 0.5

def evaluate(y_true, y_pred):
    """ Given two arrays of predictions and observed values, returns mse, r2 and percentage correct """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    perc_correct = [is_correct(x,y) for x,y in zip(y_true, y_pred)].count(True) / len(y_true)
    
    return mse, r2, perc_correct

def model_fit_ARIMA(data, order):
    """ Fit ARIMA model for some data with a specific order """
    model = ARIMA(data, order=order)
    return model.fit()

def get_train_set_set(data, perc_train = perc_train):
    """
        Given some data and the desired percentage of the trainign set, return a training set and test set.
        
        @param aray-like data           the data to divide
        @param float perc_train         the training set percentage
    """ 
    # Make training and test set
    train_size = int(len(data) * perc_train) 
    
    return data[0:train_size], data[train_size:].dropna()

def perform_ARIMA(data, pdq_order, perc_train = perc_train, figureName=None):
    """
        Perform arima on some data with a specific order and the desired training set percentage.
 
        @param aray-like data           the data
        @param (int,int,int) pqd_order  the p,d,q order for arima
        @param float perc_train         the training set percentage
        @param string figurePath        the path to save the figure of the results
        
    """
    
    # Get train and test set
    train, test = get_train_set_set(data, perc_train)

    # Train the model
    model_fit = model_fit_ARIMA(train, order=pdq_order)

    # Forecast and check accuracy
    forecast = model_fit.forecast(len(test))
    mse, r2, correct  = evaluate(test.values, forecast)
    print(f'Test MSE: {mse} \tfor {pdq_order[0]}, {pdq_order[1]} and {pdq_order[2]}')
    print(f'Test r2: {r2}')
    print(f'Test correct: {correct}')

    # Show observed values, fitted values, and forecasted values
    sns.lineplot(data=data.rename(columns={"mood": "Observed"}), palette=['#000000'], lw=0.6)
    sns.lineplot(data=model_fit.fittedvalues, dashes=True, label="Fitted") 
    sns.lineplot(x=test.index, y=forecast, label="Forecast")  
    plt.xlabel(f'Time')
    plt.ylabel(f'Mood')
    
    if figureName != None:
        plt.savefig(f'figures/{figureName}.png', dpi=300)
        
    plt.show()
        
    
def do_experiment_ARIMA_find_pdq_values(data, p_values, d_values, q_values, doPrint = True):
    """ 
        Try different p,d,q orders on a data-set to find the optimal p,d,q combination.
        
        @param aray-like data           the data to divide
        @param [integers] p_values      the p-values to try
        @param [integers] d_values      the d-values to try
        @param [integers] q_values      the q-values to try
        @param boolean doPrint          whether or not to print the result for each order
    """

    results = []
    
    # Get train and test set
    train, test = get_train_set_set(data, perc_train)

    # Information wrt sizes of train and test
    train_size_na = int(train.count())
    train_size = int(len(train))
    test_size = int(test.count())
    
    # Try out all p, d, q values
    for p in p_values:
        for d in d_values:
            for q in q_values:
                # Train the model
                model_fit = model_fit_ARIMA(train, order=(p, d, q))

                # Forecast and check accuracy
                forecast = model_fit.forecast(len(test))
                mse, r2, correct = evaluate(test.values, forecast)
                
                if doPrint:
                    print(f'MSE: {mse} \t\t\tfor {p}, {d} and {q}')
                    print(f'r2: {r2}')
                    print(f'Correct: {correct}')
                    
                    sns.lineplot(data=data)
                    sns.lineplot(data=model_fit.fittedvalues, color='red')  
                    sns.lineplot(x=test.index, y=forecast, color='orange')
                    plt.show()
                
                # Save results
                results.append([p,d,q,mse,r2,correct,train_size_na,train_size,test_size])

    return pd.DataFrame(results, columns=["p", "d", "q", "mse", "r2", "corr", "tr_na", "tr", "tst"])


# In[ ]:




