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
from earningDates import *
from datetime import datetime

eps = 1e-10
minInDay = 375

np.set_printoptions(precision = 2, suppress = True)
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',10)

debug = False

def isTradeDayStart(time):
    '''
    Returns true if the time is start of the trading day 
    '''

    currTimeStamp = datetime.fromtimestamp(time)
    currDay  = currTimeStamp.date()
    currYear = currTimeStamp.date().year
    currHour = currTimeStamp.time().hour
    currMins = currTimeStamp.time().minute

    if ((currYear == 2009) and ((currHour == 9) and (currMins == 55))):
        return True
    elif ((currYear != 2009) and ((currHour == 9) and (currMins == 15))):
        return True
    else:
        return False

# Functions to compute statistics
def avgwin(x):
    return np.mean([y for y in x if y>0])

def avgloss(x):
    return np.mean([y for y in x if y<0])

def winratio(x):
    return avgwin(x) / avgloss(x)

def accuracy(x):
    return ((100.0 * len([y for y in x if y>=0])) / (len(x) + eps))

class Analyser():

    def __init__(self, config):

        self.year       = config['YEAR']
        self.startTime  = config['START_TIME']
        self.endTime    = config['END_TIME']
        self.stockList  = config['STOCK_LIST']
        self.mode       = config['MODE']
        self.logBucket  = config['LOG_BUCKET_DATA']
        self.outlier    = config["IGNORE_OUTLIERS"]
        self.lowPerc    = config["LOW_PERCENTILE"]
        self.highPerc   = config["HIGH_PERCENTILE"]
        self.hold       = config['HOLD']
        self.pickleFlag = config['PICKLE_FLAG']
        self.volType    = config['VOL_TYPE']
        self.volMethod  = config['VOL_METHOD']
        self.gapVWAP    = config['VWAP_GAP_WINDOW']
        self.priceVWAP  = config['VWAP_PRICE_WINDOW']
        self.absFlag    = config['ABS_FLAG']

        self.hedgeFlag  = config['HEDGE']
        self.hedgeStock = config['HEDGE_STOCK']
        self.limitFlag  = config['LIMIT_FLAG']
        self.stopLoss   = config['STOP_LOSS_VAL']
        self.targetType = config['TARGET_TYPE']
        self.target     = config['TARGET_VAL']

        self.miscStatFlag = config['MISC_STATS_FLAG']
        self.notionalFlag = config['NOTIONAL_FLAG']
        self.gapFillAccFlag = config['GAP_FILL_ACCURACY']

        self.modStockList = self.stockList

        if (self.hedgeFlag):
            self.betaCorrelation = config['BETA_CORR']
            self.modStockList = [config['HEDGE_STOCK']] + self.stockList
            self.corrFlag = config['BETA_CORR_TYPE']


        # Class members containing relevant statistics
        # self.results: Dictionary containing stock names as keys
        #               Maps to a list of lists, where each list member
        #               contains gapSize, timeStamp, Open/Close prices
        #               along with holding periods, etc
        self.results            = {}
        self.gapListNormalized  = []
        self.tradeList          = {}
        self.dataList           = None
        self.timeIdx            = None

        self.bins          = config['BINS']
        self.calculateBins = config['CALCULATE_BINS']
        
        self.excludeEarnings = config['EXCLUDE_EARNINGS']

        if (self.excludeEarnings):
            self.earningTS = getEarnings(config)
            self.earningTS.getEarningTimeStamps()
            self.earningTimeStamps = self.earningTS.earningTimeStamps

        self.volumeTrendFlag = config['VOLUME_TREND']

    def isTradeDayStart(self,time):
        '''
        Returns true if the time is start of the trading day 
        '''

        currTimeStamp = datetime.fromtimestamp(time)
        currDay   = currTimeStamp.date()
        currYear  = currTimeStamp.date().year
        currMonth = currTimeStamp.date().month
        currDate  = currTimeStamp.date().day
        currHour  = currTimeStamp.time().hour
        currMins  = currTimeStamp.time().minute

        if ((currYear == 2009) and ((currHour == 9) and (currMins == 55))):
            return True
        elif((currYear == 2010) and ((currMonth < 10) or (currDate <= 17 and currMonth == 10)) and ((currHour == 9) and (currMins == 00))):
            return True
        elif((currYear == 2010) and ((currMonth >= 11) or (currDate > 17 and currMonth == 10)) and ((currHour == 9) and (currMins == 15))):
            return True
        elif ((currYear != 2009) and (currYear != 2010) and ((currHour == 9) and (currMins == 15))):
            return True
        else:
            return False

    def setPriceData(self, dataDf):
        '''
        Sets price data to the passed dictionary
        Returns:
            None, only class members are modified
        '''
        self.dataList = dataDf
        self.timeIdx  = dataDf[self.stockList[0]].index

    def getVolAvgPriceDf(self, stock, tradeIdx, left, right, notional = False):
        '''
        More optimized compared to the above function
        Computes the volume weighted price for the range [idx + left, idx + right)
        for all indices provided, price = (low + high)/2
        '''
        priceList = self.dataList[stock]
        avgPrice  = priceList['open'].iloc[tradeIdx].values * 0.0
        volSum    = avgPrice * 0.0

        for i in range(left, right):
            currPrice = (priceList['open'].iloc[tradeIdx + i].values + priceList['close'].iloc[tradeIdx + i].values) / 2.0
            # currPrice = (priceList['open'].iloc[tradeIdx + i].values + priceList['open'].iloc[tradeIdx + i].values) / 2.0
            avgPrice = avgPrice + (currPrice * priceList['vol'].iloc[tradeIdx + i].values)
            volSum = volSum + priceList['vol'].iloc[tradeIdx + i].values

        volAvgPrice = priceList['open'].iloc[tradeIdx] * 0.0
        volAvgPrice = pd.Series((avgPrice / (volSum + eps)), index = priceList['open'].iloc[tradeIdx].index)

        volAvgPrice[(tradeIdx + left) < 0] = np.nan

        if (notional):
            notionalVal = pd.Series(avgPrice, index = priceList['open'].iloc[tradeIdx].index)
            notionalVal[(tradeIdx + left) < 0] = np.nan
            return notionalVal, volAvgPrice
        else:
            return volAvgPrice

    def getMinMax(self, stock, tradeIdx, left, right):

        priceList = self.dataList[stock]
        minimum = priceList['low'].iloc[tradeIdx + left].values
        maximum = priceList['high'].iloc[tradeIdx + left].values

        for i in range(left+1,right):
            minimum = np.minimum(minimum, priceList['low'].iloc[tradeIdx + i].values)
            maximum = np.maximum(maximum, priceList['high'].iloc[tradeIdx + i].values)

        return minimum,maximum

    def getDayPriceDf(self, stock, tradeIdx, numMins, mode = 'min'):
        '''
        Returns the cumulated data in the next numMins
        mode specifies min or max
        '''

        priceList = self.dataList[stock]
        feature   = 'low' if mode == 'min' else 'high'
        offset    = 1e10 if mode == 'min' else 0
        cumPrice  = (priceList[feature].iloc[tradeIdx].values * 0.0) + offset

        for i in range(0, numMins):
            if (mode == 'min'):
                cumPrice = np.minimum.reduce([cumPrice, priceList[feature].iloc[tradeIdx + i].values])
            else:
                cumPrice = np.maximum.reduce([cumPrice, priceList[feature].iloc[tradeIdx + i].values])

        cumPrice = pd.Series(cumPrice, index = priceList[feature].iloc[tradeIdx].index)

        return cumPrice

    def getGapStats(self, holdPeriodList, verbose = True):
        '''
        Gives the statistics (Gap trading) for all hold periods specified
        The stats include,
        'timestamp', 'currOpen', 'prevClose', 'entryPrice', 'gapSize', 'dailyRet',
        'signal', 'vol', 'gapRatio'
        The object returned is a dictionary with the stocks as keys. Each containing
        a dataframe containing the above mentioned stats
        Args:
            holdPeriodList: Contains holding periods as number of minutes
            volType; dailyVol or nDayVol (n = 70, 30 respectively)
        Returns:
            Dictionary as described above
        '''

        if verbose:
            print 'Current analysis progress:'
            bar = progressbar.ProgressBar(maxval = len(self.modStockList), \
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

        # Initialize base objects
        currOpen   = {}
        prevClose  = {}
        prevDayRet = {}
        currLow    = {}
        currHigh   = {}
        entryPrice = {}
        gapSize    = {}
        dailyRet   = {}
        vol        = {}
        gapRatio   = {}
        percentile = {}
        signal     = {}
        tradeList  = {}

        minimum    = {}
        maximum    = {}
        notional   = {}
        notionalPerc     = {}
        round_pcile      = {}
        rollingSizeMean  = {}
        rollingRatioMean = {}

        volumeCurrOpen  = {}
        volumePrevClose = {}
        volumeOneMin    = {}
        volumeTwoMin    = {}
        volumeTrend     = {}

        # Get list of indices on which trades need to be done
        tradeIdx = np.where(self.timeIdx.map(self.isTradeDayStart))[0]
        
        # print self.timeIdx[tradeIdx].map(datetime.fromtimestamp)

        if (self.notionalFlag):
            notionalTot = None
            for count in range(len(self.modStockList)):
                stock = self.modStockList[count]
                notional[stock], _ = self.getVolAvgPriceDf(stock, tradeIdx, 0, 2, notional = True)
                notional[stock].fillna(eps, inplace = True)

                # Skipping the index in notional computation
                if (self.hedgeStock != stock):
                    if (notionalTot is None):
                        notionalTot = deepcopy(notional[stock])
                    else:
                        notionalTot += notional[stock]

        # Perform analysis for each stock
        for count in range(len(self.modStockList)):

            stock = self.modStockList[count]

            # Price computation by using VWAP
            currOpen[stock]   = self.getVolAvgPriceDf(stock, tradeIdx, 0, self.gapVWAP)
            prevClose[stock]  = self.getVolAvgPriceDf(stock, tradeIdx, -self.gapVWAP, 0)
            entryPrice[stock] = self.getVolAvgPriceDf(stock, tradeIdx, self.gapVWAP, self.gapVWAP + self.priceVWAP)
            gapSize[stock]    = (currOpen[stock] / (prevClose[stock] + eps)) - 1
            dailyRet[stock]   = (currOpen[stock] / (currOpen[stock].shift(1) + eps)) - 1
            signal[stock]     = -np.sign(currOpen[stock] - prevClose[stock])

            volumeCurrOpen[stock]  = self.dataList[stock]['vol'].iloc[tradeIdx].values
            volumePrevClose[stock] = self.dataList[stock]['vol'].iloc[tradeIdx - 1].values
            volumeOneMin[stock]    = self.dataList[stock]['vol'].iloc[tradeIdx + 1].values
            volumeTwoMin[stock]    = self.dataList[stock]['vol'].iloc[tradeIdx + 2].values

            volDF = pd.concat([pd.DataFrame(volumePrevClose[stock]),pd.DataFrame(volumeCurrOpen[stock]),\
                pd.DataFrame(volumeOneMin[stock]),pd.DataFrame(volumeTwoMin[stock])], axis = 1)

            volumeTrend[stock] = volDF.diff(periods = 1, axis = 1).iloc[:,1:].sum(axis = 1) / 4
            # print(volumeTrend[stock])

            if (self.volMethod == 'ema'):
                if (self.volType == 'stdVol'):
                    vol[stock] = dailyRet[stock].ewm(span = 70, min_periods = 30).std()
                else:
                    vol[stock] = gapSize[stock].ewm(span = 70, min_periods = 30).std()
            else:
                if (self.volType == 'stdVol'):
                    vol[stock] = dailyRet[stock].rolling(window = 70, min_periods = 30, center = False).std()
                else:
                    vol[stock] = gapSize[stock].rolling(window = 70, min_periods = 30, center = False).std()

            gapRatio[stock] = gapSize[stock] / (vol[stock] + eps)
            # gapRatio[stock] = gapSize[stock] * 1.0 + (0.0 * vol[stock])

            # print (gapSize[stock].values)[:20]
            # print (dailyRet[stock].values)[:20]
            # print (currOpen[stock].values)[:20]
            # print (prevClose[stock].values)[:20]
            # print (vol[stock].values)

            # vol[stock].plot()
            # tmp = deepcopy(gapSize[stock])
            # tmp.index = pd.to_datetime(tmp.index, unit = 's')
            # tmp.plot()
            # plt.ylim((-0.12,0.12))
            # plt.ylabel('gapSize')
            # (gapRatio[stock] / 300.0).plot()
            # plt.show()

            # tmp = deepcopy(gapRatio[stock])
            # tmp.index = pd.to_datetime(tmp.index, unit = 's')
            # tmp.plot()
            # plt.ylim((-6,6))
            # plt.ylabel('gapRatio')

            if (self.absFlag):
                gapRatio[stock] = np.abs(gapRatio[stock])

            # rollingSizeMean[stock] = (gapSize[stock]).rolling(window = 70, min_periods = 30).apply(lambda x: pd.Series(x).mean())
            # rollingRatioMean[stock] = (gapRatio[stock]).rolling(window = 70, min_periods = 30).apply(lambda x: pd.Series(x).mean())
            percentile[stock]  = (gapRatio[stock]).rolling(window = 70, min_periods = 30).apply(lambda x: pd.Series(x).rank(pct = True).iloc[-1])
            percentile[stock]  *= 100.0
            round_pcile[stock] = percentile[stock].dropna().apply(lambda x: int((x / 20.0) + 0.5) * 20)

            tradeList[stock] = pd.concat([currOpen[stock], prevClose[stock], entryPrice[stock],\
                                          gapSize[stock], dailyRet[stock], signal[stock],\
                                          vol[stock], gapRatio[stock], percentile[stock], round_pcile[stock]],\
                                          # rollingSizeMean[stock], rollingRatioMean[stock]],\
                                          keys = ['currOpen', 'prevClose', 'entryPrice', 'gapSize',\
                                          'dailyRet', 'signal', 'vol', 'gapRatio', 'percentile', 'round_pcile'], axis = 1)
            tradeList[stock]['stockName'] = stock
            tradeList[stock]['binID'] = pd.cut(tradeList[stock]['gapRatio'], self.bins, right = False, labels = range(1, len(self.bins)))
            
            if (self.miscStatFlag):
                currLow[stock]  = self.getDayPriceDf(stock, tradeIdx, numMins = minInDay, mode = 'min')
                currHigh[stock] = self.getDayPriceDf(stock, tradeIdx, numMins = minInDay, mode = 'max')
                # tradeList[stock]['openInLowHigh'] = ~ ((currLow[stock].shift(1) <= currOpen[stock]) & (currOpen[stock] <= currHigh[stock].shift(1)))
                # tradeList[stock]['openInLowHigh'] = (-1 + 2 * (tradeList[stock]['openInLowHigh'])) \
                # * pd.concat([np.abs(1 - (currLow[stock].shift(1) / currOpen[stock])),\
                #np.abs(1 - (currHigh[stock].shift(1) / currOpen[stock]))], axis = 1).min(axis = 1)

                tradeList[stock]['isInsideGap'] = (currHigh[stock].shift(1) > currOpen[stock]) & (currOpen[stock] > currLow[stock].shift(1))
                tradeList[stock]['openInLowHigh'] = (currOpen[stock] - currLow[stock].shift(1)) / (currHigh[stock].shift(1) - currLow[stock].shift(1))

                tradeList[stock]['currLow'] = currLow[stock]
                tradeList[stock]['currHigh'] = currHigh[stock]

            if (self.notionalFlag):
                notional[stock]     = notional[stock] / notionalTot
                notionalPerc[stock] = (notional[stock]).rolling(window = 70, min_periods = 30).apply(lambda x: pd.Series(x).rank(pct = True).iloc[-1])
                notionalPerc[stock] *= 100.0
                tradeList[stock]['notionalPerc'] = notionalPerc[stock]

            # Computing n-day returns
            for n in range(1, 5):
                tradeList[stock]['prev' + str(n) +'DayRet'] = (prevClose[stock] / (currOpen[stock].shift(n) + eps)) - 1

            # Storing gap of the index
            if (self.hedgeFlag):
                tradeList[stock]['gapRatioIndex'] = tradeList[self.hedgeStock]['gapRatio']

            # Computing beta for each day
            if (self.hedgeFlag):
                if (self.corrFlag != 'constant'):
                    self.betaCorrelation = pd.rolling_corr(dailyRet[stock], dailyRet[self.hedgeStock], window = 70)
                beta = self.betaCorrelation * (vol[stock] / (vol[self.hedgeStock] + eps))
                tradeList[stock]['beta'] = beta

            if (self.limitFlag):
                limitHit = (currOpen[stock] * 0.0)

            if (self.gapFillAccFlag):
                gapFill       = (currOpen[stock] * 0.0)
                gapFillPeriod = (currOpen[stock] * 0.0)

            for i in range(len(holdPeriodList)):

                hold = holdPeriodList[i]

                finClose = self.getVolAvgPriceDf(stock, tradeIdx, hold + self.gapVWAP, hold + self.gapVWAP + self.priceVWAP)
                profit = signal[stock] * ((finClose / (entryPrice[stock] + eps)) - 1)

                if (self.hedgeFlag):
                    finCloseIndex = self.getVolAvgPriceDf(self.hedgeStock, tradeIdx, hold + self.gapVWAP, hold + self.gapVWAP + self.priceVWAP)
                    profitIndex = (beta) * signal[stock] * ((finCloseIndex / (entryPrice[self.hedgeStock] + eps)) - 1)
                    profit -= profitIndex

                profitPerVol = profit / ((vol[stock] + eps) * np.sqrt(hold * 1.0 / minInDay))

                tradeList[stock]['finClose' + '_' + str(hold)] = finClose
                tradeList[stock]['profit' + '_' + str(hold)] = profit
                tradeList[stock]['profitPerVol' + '_' + str(hold)] = profitPerVol

                if (self.gapFillAccFlag):
                    gapFillPeriod = ((1.0 - gapFill) * hold) + (gapFill * gapFillPeriod)
                    gapFill[(signal[stock] * (finClose - prevClose[stock])) > 0.0] = 1.0

                if (self.limitFlag):

                    prevHold = holdPeriodList[max(0, i-1)]

                    if (self.targetType == 'prevClose'):
                        target = np.abs((prevClose[stock] / (entryPrice[stock] + eps)) - 1)
                    elif (self.targetType == 'constant'):
                        target = self.target
                    else:
                        target = (1.0 / eps)

                    stopLoss = self.stopLoss

                    lowVal = (profit * 0.0) + stopLoss
                    tradeList[stock]['low_val' + '_' + str(hold)] = lowVal
                    highVal = (profit * 0.0) + target
                    tradeList[stock]['high_val' + '_' + str(hold)] = highVal

                    tradeList[stock]['limit_hold' + '_' + str(hold)] = (finClose * 0.0)
                    tradeList[stock]['limit_hold' + '_' + str(hold)] = \
                                    ((1.0 - limitHit) * hold) + (limitHit * tradeList[stock]['limit_hold' + '_' + str(prevHold)])

                    tradeList[stock]['limit_finClose' + '_' + str(hold)] = (finClose * 0.0)
                    tradeList[stock]['limit_finClose' + '_' + str(hold)] = \
                                    ((1.0 - limitHit) * finClose) + (limitHit * tradeList[stock]['limit_finClose' + '_' + str(prevHold)])

                    tradeList[stock]['limit_profit' + '_' + str(hold)] = (finClose * 0.0)
                    tradeList[stock]['limit_profit' + '_' + str(hold)] = \
                                    ((1.0 - limitHit) * profit) + (limitHit * tradeList[stock]['limit_profit' + '_' + str(prevHold)])

                    limitHit[~ ((lowVal <= profit) & (profit <= highVal))] = 1.0

            if (self.gapFillAccFlag):
                tradeList[stock]['gapFill_period'] = gapFillPeriod

            if(self.volumeTrendFlag):
                tradeList[stock]['volumePrevClose'] = volumePrevClose[stock]
                tradeList[stock]['volumeCurrOpen']  = volumeCurrOpen[stock]
                tradeList[stock]['volumeOneMin']    = volumeOneMin[stock]
                tradeList[stock]['volumeTwoMin']    = volumeTwoMin[stock]

            tradeList[stock].dropna(axis = 0, how = 'any', inplace = True)
            
            if (self.excludeEarnings):
                earningTSes = self.earningTimeStamps[stock]
                earningTSes = list(set(earningTSes) & set(tradeList[stock].index))
                tradeList[stock].drop(earningTSes,inplace = True)

            if (verbose):
                bar.update(count)

        if (verbose):
            bar.finish()

        # plt.show()

        #Functionality to remove outlier, triggered by the outlier flag in config
        if (self.outlier):
            profitDf = pd.DataFrame()
            for stock in self.modStockList:
                profitDf = profitDf.append(tradeList[stock])
            lowLim = profitDf['profit_' + str(self.hold)].quantile(self.lowPerc)
            highLim = profitDf['profit_' + str(self.hold)].quantile(self.highPerc)

            for stock in self.modStockList:
                tradeList[stock] = tradeList[stock][tradeList[stock]['profit_' + str(self.hold)] < highLim]
                tradeList[stock] = tradeList[stock][tradeList[stock]['profit_' + str(self.hold)] > lowLim]

        self.tradeList = tradeList

        if (self.pickleFlag):
            # suffix = 'test'
            # suffix = 'min_1_stopLoss_20bp_full_360_noOutlier'
            # suffix = 'noStopLoss_partial_360_noOutlier_newVol'
            suffix = 'noStopLoss_full_360_noOutlier_newVol'
            directory = 'results/' + suffix + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(directory + 'gapStatsDf_' + str(self.year) + '_' + suffix + '.pickle','wb') as handle:
                pickle.dump(tradeList, handle, protocol = pickle.HIGHEST_PROTOCOL)

        return tradeList

    def printStats(self, holdPeriodList, globalTradeList = None, globalFlag = False, verbose = False):

        if (globalFlag):
            print (' '.join(['*']*50))
            print 'Global Statistics: Cumulative over all years'
            print (' '.join(['*']*50))

        cumTrades = pd.DataFrame()
        if (not globalFlag):
            for key in self.tradeList.keys():
                cumTrades = cumTrades.append(pd.DataFrame(self.tradeList[key]))
        else:
            for key in globalTradeList.keys():
                curr           = pd.DataFrame(globalTradeList[key])
                curr['ticker'] = key
                cumTrades    = cumTrades.append(curr)

        bucket = 'binID'
        statNames = ['count', 'acc', 'max', 'min', 'std', 'avgwin', 'avgloss', 'winratio', 'exp']
        funcList = [len, accuracy, np.max, np.min, np.std, avgwin, avgloss, winratio, np.mean]

        currStats = {}
        currStatsWithLimit = {}

        for hold in holdPeriodList:

            profitCol  = 'profit_' + str(hold)

            currStats[hold] = pivot_table(cumTrades, values = [profitCol], index = [bucket], aggfunc = funcList)
            currStats[hold].columns = [s1 + ':' + s2.split('_')[-1] for (s1, s2) in currStats[hold].columns.tolist()]
            if (self.gapFillAccFlag):
                cumTrades['gapFill'] = hold - (cumTrades['gapFill_period'] + 0.1)
                currStats[hold]['gapFillAcc:' + str(hold)] = pivot_table(cumTrades, values = ['gapFill'], index = [bucket], aggfunc = accuracy)

            if (verbose):
                print (' '.join(['*']*50))
                print 'Hedging:', self.hedgeFlag, "| Without Limits |", "Hold", hold
                print (' '.join(['*']*50))
                print currStats[hold]

            if (self.limitFlag):
                currStatsWithLimit[hold] = pivot_table(cumTrades, values = ['limit_' + profitCol], index = [bucket], aggfunc = funcList)
                currStatsWithLimit[hold].columns = [s1 + ':' + s2.split('_')[-1] for (s1, s2) in currStatsWithLimit[hold].columns.tolist()]
            
                if (verbose):
                    print (' '.join(['*']*50))
                    print 'Hedging:', self.hedgeFlag, "| With Limits:", self.stopLoss, 'Target:', self.targetType, "| Hold", hold
                    print (' '.join(['*']*50))
                    print currStatsWithLimit[hold]

        print 'Profit average is:', cumTrades['profit_' + str(self.hold)].mean()

        if (self.gapFillAccFlag):
            print 'Cumulative Results: Gap Fill Probability'
            print pd.concat([currStats[hold]['gapFillAcc:' + str(hold)] for hold in holdPeriodList], axis = 1)

        print 'Cumulative Results: Without Limits'
        print pd.concat([currStats[hold]['accuracy:' + str(hold)] for hold in holdPeriodList], axis = 1)
        print pd.concat([currStats[hold]['mean:' + str(hold)] for hold in holdPeriodList], axis = 1)

        if (self.limitFlag):

            print 'Cumulative Results: With Limits =', self.stopLoss, 'Target:', self.targetType
            print pd.concat([currStatsWithLimit[hold]['accuracy:' + str(hold)] for hold in holdPeriodList], axis = 1)
            print pd.concat([currStatsWithLimit[hold]['mean:' + str(hold)] for hold in holdPeriodList], axis = 1)

        print("Gap Bucket Statistics\n")
        print(pivot_table(cumTrades, values = ['gapSize', 'gapRatio'], index = [bucket], aggfunc = [len, np.max, np.min, np.std, np.mean]))
        # print(cumTrades['gapRatio'].describe([0.01, 0.1, 0.17, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.67, 0.7, 0.8, 0.83, 0.9, 0.99]))
        # print(cumTrades['gapSize'].describe([0.01, 0.1, 0.17, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.67, 0.7, 0.8, 0.83, 0.9, 0.99]))
