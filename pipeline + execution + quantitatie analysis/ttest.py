from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
from sortedcontainers import SortedList
from operator import truediv
from scipy.stats import ttest_ind
from bisect import bisect_left, bisect
import collections
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from pandas import pivot_table
import time
import pickle
import progressbar
import os.path
from copy import deepcopy
from scipy import stats
import functools
import matplotlib.pyplot as plt

eps = 1e-10

np.set_printoptions(precision = 2, suppress = True)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 30)

def loadAnalyserResults(yearList, suffix = 'noStopLoss_partial_360_noOutlier', pricePerc = True):
    '''
    Takes a list of years as input and loads the
    result files into it.
    '''

    yearStats = {}
    # suffix = 'min_1_stopLoss_20bp_full_360'
    # suffix = 'noStopLoss_partial_360'
    for year in yearList:
        print 'Currently at year:', year
        with open('results/' + suffix + '/' + 'gapStatsDf_' + str(year) + '_' + suffix + '.pickle', 'rb') as handle:
            yearStats[year] = pickle.load(handle)

    # Preprocessing to add desired stats
    for year in yearList:
        stockList = yearStats[year].keys()
        for stock in stockList:
            yearStats[year][stock]['prevLow']  = yearStats[year][stock]['currLow'].shift(1)
            yearStats[year][stock]['prevHigh'] = yearStats[year][stock]['currHigh'].shift(1)
            yearStats[year][stock]['prevOpen'] = yearStats[year][stock]['currOpen'].shift(1)
            yearStats[year][stock].dropna(axis = 0, how = 'any', inplace = True)

    stats = yearStats[yearList[0]]
    stockList = stats.keys()
    for year in yearList[1:]:
        for stock in stockList:
            stats[stock] = stats[stock].append(yearStats[year][stock])

    grandStats = pd.DataFrame()
    for stock in stockList:
        tmpDf = pd.DataFrame()
        tmpDf = tmpDf.append(stats[stock])
        tmpDf['stockName'] = stock

        if (pricePerc):
            tmpDf['pricePerc'] = (tmpDf['currOpen']).rolling(window = 20).apply(lambda x: pd.Series(x).rank(pct = True).iloc[-1])
            tmpDf['pricePerc'] *= 100.0
            tmpDf.dropna(inplace = True)

        grandStats = grandStats.append(deepcopy(tmpDf))

    print 'Loading results complete!'

    return grandStats

def getHoldWiseStats(grandStats, holdList, colList):

    holdGrandStats = pd.DataFrame()
    for hold in holdList:
        print 'Currently at holding period:', hold
        tmpDf = grandStats.loc[:,colList]
        tmpDf['profit'] = grandStats.loc[:,'profit_' + str(hold)]
        tmpDf['hold'] = hold
        holdGrandStats = holdGrandStats.append(deepcopy(tmpDf))

    print 'Getting hold wise stats complete!'
    return holdGrandStats

def extractWeekDayFromTime(time):
    timeStamp = datetime.fromtimestamp(time)
    weekDay   = timeStamp.weekday()
    return (weekDay)

def extractGapZone(row):

    result = 0

    opens = row['prevOpen'] 
    close = row['prevClose']
    low   = row['prevLow']
    high  = row['prevHigh']
    
    if (opens < close):
        if (row['currOpen'] <= low):
            result = 1
        elif (low <= row['currOpen'] <= opens):
            result = 2
        elif (opens <= row['currOpen'] <= close):
            result = 3
        elif (close <= row['currOpen'] <= high):
            result = 4
        else:
            result = 5
    else:
        if (row['currOpen'] <= low):
            result = 1
        elif (low <= row['currOpen'] <= close):
            result = 2
        elif (close <= row['currOpen'] <= opens):
            result = 3
        elif (opens <= row['currOpen'] <= high):
            result = 4
        else:
            result = 5

    return result

def getWeekDay(grandStats):
    return grandStats.index.map(extractWeekDayFromTime)

def getGapZone(grandStats):
    return grandStats.apply(extractGapZone, axis = 1)

def getFetRank(fet, grandStats):
    return grandStats.groupby(grandStats.index)[fet].rank(pct = True) * 100.0

def avgwin(x):
    return np.mean([y for y in x if y>0])

def avgloss(x):
    return np.mean([y for y in x if y<0])

def winratio(x):
    return np.abs(avgwin(x) / avgloss(x))

def accuracy(x):
    return ((100.0 * len([y for y in x if y>=0])) / (len(x) + eps))

def getStats(profit):
    '''
    Get mentioned stats for a passed dataframe i.e. profit
    '''

    results = {}
    results['count'] = len(profit)
    results['acc'] = accuracy(profit)
    results['max'] = profit.max()
    results['min'] = profit.min()
    results['winratio'] = winratio(profit)
    results['avgwin'] = avgwin(profit)
    results['avgloss'] = avgloss(profit)
    results['exp'] = np.mean(profit)

    return results

def plotDist(test, title, boundary):
    """
    Functionality to plot the distribution of profits to check if its normal
    """

    max = np.amax(test['profit'])
    min = np.amin(test['profit'])
    histo = np.histogram(test['profit'], bins = 100, range = (min,max))
    freqs = histo[0]
    rangebins = (max - min)
    numberbins = (len(histo[1])-1)
    interval = (rangebins/numberbins)
    newbins = np.arange((min), (max), interval)
    histogram = plt.bar(newbins, freqs, width=0.001)
    plt.title(title + ' ' + str(boundary))
    plt.show()

def performTTest(tradeDf, feature, cond, hold, plotDistFlag = False):
    '''
    Performs the t test and returns the results,
    feature represents the attribute to be considered,
    It is actually a dataframe of the concerned values
    cond is a function which divides the set into two
    splits
    '''

    results = {}

    posSample = tradeDf[feature.map(cond)]
    negSample = tradeDf[~ feature.map(cond)]

    if(plotDistFlag):
        plotDist(posSample, "posSample", cond.keywords['boundary'])
        plotDist(negSample, "negSample", cond.keywords['boundary'])

    results['pos'] = getStats(posSample['profit'])
    results['neg'] = getStats(negSample['profit'])
    results['tot'] = getStats(tradeDf['profit'])

    # print(len(posSample[posSample['gapFilled' + '_' + str(hold)] == True]) / float(len(posSample)))
    # print(len(negSample[negSample['gapFilled' + '_' + str(hold)] == True]) / float(len(negSample)))
    # print(results['pos']['gapFilled'], results['neg']['gapFilled'])

    # results = pd.DataFrame(results).transpose()
    ttestResult = stats.ttest_ind(posSample['profit'], negSample['profit'])

    return results, ttestResult

def getMultiIndexDataFrame(dictionary, levels):
    '''
    Get multi index dataframe from dictionary with 'levels' level of keys
    '''

    if (levels == 1):
        if isinstance(dictionary, pd.DataFrame):
            return dictionary
        return pd.DataFrame(dictionary, index = [''])

    keyList, frameList = [], []
    for key in sorted(dictionary.keys()):
        keyList.append(key)
        frameList.append(getMultiIndexDataFrame(dictionary[key], levels - 1))
    tmpDf = pd.concat(frameList, keys = keyList)

    return tmpDf

def baseFilter(x, boundary):
    if (x > boundary):
        return True
    else:
        return False

def rangeFilter(x, left, right):
    if (left <= x < right):
        return True
    else:
        return False

def eqFilter(x, value):
    if (x == value):
        return True
    else:
        return False

if __name__ == '__main__':

    # yearList = [2016]
    yearList = [2010,2012,2014,2016]
    # holdList = (range(3, 30, 3) + range(30, 120, 6) + range(120, 301, 12))
    holdList = [60,120,240,360]
    # holdList = [60]
    # Compute daily gap Rank

    weekDay = False
    gapZone = False
    gapRank = False
    volRank = True
    rangeFlag     = True
    volumeRank    = True
    openInLowHigh = True 

    colList = [u'currOpen', u'prevClose', u'entryPrice', u'gapSize', u'dailyRet',
                u'signal', u'vol', u'gapRatio', u'percentile', u'round_pcile',
                u'rollingSizeMean', u'rollingRatioMean', u'stockName', u'beta',
                u'gapRatioIndex', u'prevLow', u'prevHigh', u'prevOpen', u'notionalPerc',
                u'pricePerc']

    grandStats = loadAnalyserResults(yearList, suffix = 'noStopLoss_full_360_noOutlier_newVol')

    # for hold in holdList:
    #     colList.append(u'gapFilled'+ '_' + str(hold))

    # print(colList)
    # grandStats = loadAnalyserResults(yearList)

    # print grandStats.columns
    # grandStats = grandStats[np.abs(grandStats['gapRatio']) > 0.1]
    # grandStats = grandStats[grandStats['gapRatio'] < 0]

    if (weekDay):
        grandStats['weekDay'] = getWeekDay(grandStats)
        colList.append('weekDay')

    if (gapZone):
        grandStats = grandStats[grandStats['prev3DayRet'] < 0]
        grandStats['gapZone'] = getGapZone(grandStats)
        colList.append('gapZone')

    if (gapRank):
        grandStats['gapRatioZ'] = (grandStats['gapRatio'] - grandStats['gapRatioIndex'])
        # grandStats['gapRatioZ'] = np.abs(grandStats['gapRatioZ'])
        colList.append('gapRatioZ')
        grandStats = grandStats.sort_index()
        grandStats['gapDailyRank'] = getFetRank('gapRatio', grandStats)
        colList.append('gapDailyRank')

    if (volRank):
        grandStats = grandStats.sort_index()
        grandStats['volDailyRank'] = getFetRank('vol', grandStats)
        colList.append('volDailyRank')

    if (rangeFlag):
        grandStats['openInLowHighInt'] = grandStats['openInLowHigh'] * 1.0
        colList.append('openInLowHighInt')

    if(volumeRank):
        grandStats = grandStats.sort_index()
        grandStats['volumeDailyRank'] = getFetRank('volumeCurrOpen', grandStats)
        colList.append('volumeDailyRank')    
        
    # print set(grandStats['weekDay'].values)
    fetStats = (grandStats['gapRatio']).describe()
    print fetStats

    # fetStats.to_clipboard()
    # input('completed feature description')

    holdGrandStats = getHoldWiseStats(grandStats, holdList, colList)

    statList = {}
    ttestList = {}

    for hold in holdList:
        statList[hold] = {}
        ttestList[hold] = {}

        tradeDf = holdGrandStats[holdGrandStats['hold'] == hold]
        # print(tradeDf)

        print(tradeDf['vol'].describe())

        # plt.scatter(tradeDf['profit'], tradeDf['volDailyRank'], marker = 'x')
        # plt.show()

        # paramList = [-0.03, -0.01, -0.0025, -0.001, 0, 0.001, 0.0025, 0.0075]
        # paramList = [1, 2, 3, 4, 5]
        # paramList = [-3, -1, 0, 1, 3]
        # paramList = [-1, -0.5, 0, 0.5, 1]
        # paramList = [0.1, 0.25, 0.5, 1, 3]
        # paramList = [0.001, 0.005, 0.0075, 0.01, 0.025]
        # paramList = [10, 30, 50, 70, 90]
        # paramList = [-0.03,-0.015,-0.01,0,0.01]
        paramList = [0,0.2,0.4,0.6,0.8,1]

        for i in range(len(paramList)):
            param = paramList[i]
            print 'Currently on', param
            # Constructs a partial function, freezing the specified params
            # partialCond = functools.partial(baseFilter, boundary = param)
            partialCond = functools.partial(rangeFilter, left = paramList[i], right = paramList[min(i+1, len(paramList) - 1)])
            # partialCond = functools.partial(eqFilter, value = param)
            # feature = tradeDf['gapRatio']
            feature = tradeDf['openInLowHighInt']

            # feature = tradeDf['volumeDailyRank']
            # feature = tradeDf['volDailyRank']
            # feature = tradeDf['notionalPerc']
            # feature = tradeDf['pricePerc']
            # feature = tradeDf['gapRatio']
            # feature = tradeDf['percentile']
            results, ttestResult = performTTest(tradeDf, feature, partialCond, hold)
            statList[hold][param] = results
            ttestList[hold][param] = {}
            ttestList[hold][param][0] = {'statistic': ttestResult[0], 'p-value': ttestResult[1]}
            ttestList[hold][param][1] = {'statistic': ttestResult[0], 'p-value': ttestResult[1]}
            ttestList[hold][param][2] = {'statistic': ttestResult[0], 'p-value': ttestResult[1]}

    resultColList = ['count', 'acc', 'gapFilled', 'max', 'min', 'winratio', 'avgwin', 'avgloss', 'exp']

    statDf = getMultiIndexDataFrame(statList, levels = 4)
    statDf = statDf.loc[:, resultColList]
    print statDf
    statDf.to_clipboard()
    raw_input('copied to clipboard..')
    statDf = getMultiIndexDataFrame(ttestList, levels = 4)
    print statDf
    statDf.to_clipboard()
    raw_input('copied to clipboard..')

