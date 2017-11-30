import sys
import numpy as np
import scipy
import pandas as pd
import csv
import time
from sklearn import datasets, linear_model
import matplotlib as plt
from datetime import datetime

dailyTradeSec = 375 * 60
debug = False


def dt2ut(dt):
    epoch = pd.to_datetime('1970-01-01')
    return (dt - epoch).total_seconds()

def get_Total_PnL(df):
    #calculating net Profit or Loss
    return float(df['value'].iloc[-1]) - float(df['value'].iloc[0])

def getAnnualRetDev(df, logPeriod):
    #Calculating the standard deviation of the returns
    return np.std(list(((df['value'] / df['value'].shift(1)) - 1)[1:])) * np.sqrt(252.0 * dailyTradeSec / logPeriod)

def getMaxDD(df):
    currMax = 0
    maxDD  = 0
    for i in range(df['value'].shape[0]):
        if (currMax < df['value'].iloc[i]):
            currMax = df['value'].iloc[i]
        dd = currMax - df['value'].iloc[i]
        if (maxDD < (dd / currMax)):
            maxDD = (dd / currMax)
    return maxDD

def getAnnualRet(df, noOfDays):
    return_ratio = float(df['value'].iloc[-1]) / float(df['value'].iloc[0])
    annual_return = (((return_ratio) ** (1 / (noOfDays / 365.0))) - 1.0)
    return annual_return

def getSharpe(df, noOfDays, logPeriod):
    # Calculate and return the Sharpe Ratio
    annualRet = getAnnualRet(df, noOfDays)
    annualDev = getAnnualRetDev(df, logPeriod)
    sharpe = (annualRet / annualDev)
    return sharpe

def getCalmar(df, noOfDays):
    annualRet = getAnnualRet(df, noOfDays)
    dd = getMaxDD(df)
    return (annualRet / dd)

def get_Max_Drawdown(df):
	#calculating the maximum drawdown
	
	i = np.argmax(np.maximum.accumulate(df['value']) - df['value']) # end of the period
	j = np.argmax(df['value'][:i]) # start of period
	
	return (int(df['value'].iloc[i]) - int(df['value'].iloc[j]))

def calmar(df, totalPnL, drawdown,noOfDays):
	
	returnRatio = float(df['value'].iloc[-1]) / float(df['value'].iloc[0])
	
	if(debug):
		print (return_ratio)
	
	annualReturnPct = getAnnualRet(returnRatio,noOfDays)
	absoluteAnnualReturn = annualReturnPct * float(df['value'].iloc[0])
	
	if(debug):
		print (((returnRatio) ** (1 / (noOfDays / 365.0))) - 1.0)
		print (absoluteAnnualReturn)
	
	return absoluteAnnualReturn / np.abs(drawdown)

def get_standard_ratios(file, config):
    #opening the file and reading the content and storing it in a pandas data frame

    df = pd.read_csv(file)
    df1 = df.copy()

    df1 = df1.apply(pd.to_numeric,errors = 'ignore')
    df = df.apply(pd.to_numeric,errors = 'ignore')
    returns = df1['value'].diff()[1:]
    dailyPnL = returns
    daily_return_percent = dailyPnL
    df['time'] = df['time'].map(lambda t: time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t)))

    year = df['time'].iloc[0].split(" ")[3] + '/'
    type = "Adjusted_FUT_Data/"

    noOfDays = config['NUM_DAYS']
    logPeriod = config['LOG_PERIOD']
    # noOfDays = (datetime.fromtimestamp(config['END_TIME']) - datetime.fromtimestamp(config['START_TIME'])).days
    # noOfDays = 365

    stats = {}

    stats['PnL'] = get_Total_PnL(df)

    if(len(df) > 1):
        # stats['maxDrawdown'] = get_Max_Drawdown(df)
        stats['annualRet']   = getAnnualRet(df, noOfDays)
        stats['maxDrawdown'] = getMaxDD(df)
        stats['AnnualizedStdDev'] = getAnnualRetDev(df, logPeriod)
        # stats['CalmarRatio'] = calmar(df, stats['PnL'], stats['maxDrawdown'], noOfDays)
        stats['CalmarRatio'] = getCalmar(df, noOfDays)
        stats['Sharpe']      = getSharpe(df, noOfDays, logPeriod)
        
        print ("Total PnL: " + str(stats['PnL']))
        print ("Annual Ret: " + str(stats['annualRet']))
        print ("maxDrawdown: " + str(stats['maxDrawdown']))
        print ("StdDev: " + str(stats['AnnualizedStdDev']))
        print ("Calmar Ratio: " + str(stats['CalmarRatio']))
        print ("Sharpe Ratio: " + str(stats['Sharpe']))
    
    else:
        #Case for no trades
        print "Total PnL: 0"
        print "Max Drawdown: 0"

    return stats

if __name__ == '__main__':

    config = {}
    config['START_TIME'] = dt2ut(pd.to_datetime("2014/01/01"))
    config['END_TIME']   = dt2ut(pd.to_datetime("2014/12/31"))
    config['NUM_DAYS']   = (datetime.fromtimestamp(config['END_TIME']) - datetime.fromtimestamp(config['START_TIME'])).days
    config['LOG_PERIOD'] = 3600

    get_standard_ratios(sys.argv[1], config)
