from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from sortedcontainers import SortedList
from bisect import bisect_left, bisect
from analyser import *
from profilehooks import profile
import sys

debug = False
eps = 1e-10
tradeRatio = 7.0 / 24.0
secInDay = 86400 * tradeRatio
blockLen = 60
jumpSize = int(secInDay / blockLen)

class Strategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def generateSignal(self, input):
        """Implement signal generation method"""
        raise NotImplementedError("generateSignals() not implemented!")

class MovingAvg(Strategy):

    def __init__(self):
        # Not implemented yet
        print 'MovingAvg strategy: Did Nothing...'

    def generateSignal(self, backData, currDay):
        # backData should only contain data before the currDay

        order = pd.DataFrame(0, index = backData['open'].columns, columns = ['signal', 'qty', 'position'])

        period1 = 90
        period2 = 30

        # Only taking a look till the current day
        marketsClose = backData['close'][:currDay]
        marketOpen = backData['open'][:currDay]

        avg_p1 = marketsClose[-period1 : ].sum() / period1
        avg_p2 = marketsClose[-period2 : ].sum() / period2

        difference = avg_p1 - avg_p2
        deviation = difference.copy()
        totalDev = np.absolute(deviation).sum()

        if (totalDev == 0):
            return order
        else:
            order['signal']   = np.sign(deviation)
            # Ad-Hoc qty, just for experiments
            order['qty']      = np.absolute(deviation/totalDev)
            order['position'] = (deviation/totalDev)

        return order

class SimpleMom(Strategy):

    def __init__(self):
        # Not implemented yet
        print 'Did Nothing...'

    def generateSignal(self, backData, currDay):

        threshold = 0.0025
        window = 1

        close = backData['close'][currDay-window:currDay]
        opens = backData['open'][currDay-window:currDay]
        high = backData['high'][currDay-window:currDay]
        low = backData['low'][currDay-window:currDay]

        finVal = -1 * ((close - opens) / ((high - low) + 0.001)).iloc[-1]

        finVal[np.abs(finVal) < threshold] = 0.0
        finVal[np.abs(finVal) > 5.0] = np.sign(finVal) * 5.0

        order = pd.DataFrame(0, index = backData['open'].columns, columns = ['signal', 'qty', 'position'])
        order['signal'] = np.sign(finVal)
        order['qty'] = np.abs(finVal)
        order['position'] = finVal
        
        return order

class SimpleVol(Strategy):

    def __init__(self, config):
        self.volLookback  = config['VOL_LOOKBACK']
        self.retPeriod    = config['RET_PERIOD']
        self.stdDevMethod = config['STD_DEV_METHOD']
        self.lag          = config['LAG']

        # self.volLookback = volLookback
        # self.retPeriod = retPeriod
        # self.stdDevMethod = stdDevMethod
        # self.lag = lag

    def getReturn(self, period, opens):
        # Percentage based return for now
        # return (opens.iloc[-1] / opens.iloc[-period]) - 1
        return (opens.iloc[-1] - opens.iloc[-period])

    def getRollVol(self, period, opens):
        # Returns volatility based on open prices

        openSize = opens.shape[0]
        windowData = opens[openSize - period - self.lag: openSize - self.lag]
        retData = (windowData / windowData.shift(1)) - 1
        # retData = (windowData - windowData.shift(1))

        # print windowData
        # print retData

        if (self.stdDevMethod == "EMA"):
            retData = retData ** 2
            ewmRet = retData.ewm(span = period).mean()
            rollVol = np.sqrt(ewmRet)

            if (debug):
                print 'Before', retData
                print 'EWM', ewmRet.iloc[-1]
                print 'After ema:', rollVol.iloc[-1]
        else:
            rollVol = np.std(retData)

        # print rollVol
        # return rollVol.iloc[-1]
        return rollVol

    def generateSignal(self, backData, currDay):

        order = pd.DataFrame(0, index = backData['open'].columns, columns = ['signal', 'qty', 'position'])
        opens = backData['open'][:currDay]
        close = backData['close'][:currDay]

        retVal = self.getReturn(self.retPeriod, opens)
        rollVol = self.getRollVol(self.volLookback, opens)

        alpha = (retVal / (rollVol + eps))
        alphaNorm = alpha - np.mean(alpha)
        alphaNorm = - alphaNorm

        # print 'retval:', np.array(retVal)[:5]
        # print 'vol:', np.array(rollVol)[:5]
        # print 'alpha:', np.array(alphaNorm[:5])

        # Normalizing the positions
        alphaNorm = alphaNorm / (np.abs(alphaNorm).sum())

        # print 'alpha:', np.array(alphaNorm[:5])

        # NOTE: Destroy's the beta constant assumption
        # alphaNorm[np.abs(alphaNorm) > 1.0] = np.sign(alphaNorm) * 1.0

        if (debug):
            print 'retVal:', retVal
            print 'rollVol:', rollVol
            print 'alpha:', alphaNorm

        order['signal'] = np.sign(alphaNorm)
        order['qty'] = np.abs(alphaNorm)
        order['position'] = alphaNorm

        return order

class SimpleGapFade(Strategy):

    def __init__(self, config):
        self.window = config['WINDOW']
        self.index  = config['INDEX']
        self.hedge  = config['HEDGE_METHOD']
        self.percFlag = config['PERC_FLAG']
        self.stopLoss = config['STOP_LOSS_FLAG']
        self.stopVal  = config['STOP_LOSS_VAL']

        if (self.percFlag):
            # Setting up the percentile objects
            self.absFlag     = config['ABS_FLAG']
            self.winSize     = config['PERC_WINDOW']
            self.stockList   = config['STOCK_LIST']
            self.threshold   = config['PERC_THRESHOLD']

            self.currSamples = 0
            self.gapQueue    = {}
            self.orderedGaps = {}

            for stock in self.stockList:
                self.gapQueue[stock]    = deque([], maxlen = self.winSize) 
                self.orderedGaps[stock] = SortedList(load = 10)


    def getRollVol(self, opens, period):
        # Returns volatility based on open prices
        openSize = opens.shape[0]
        windowData = opens[openSize - (period*375): openSize]

        retData = ((windowData + eps) / (windowData.shift(1) + eps)) - 1
        rollVol = np.std(retData)

        return rollVol

    def getPercentile(self, gapSize):
        # Returns a dataframe containing their percentiles (in 0-1)
        perc = gapSize * 0.0

        for stock, gap in gapSize.iteritems():
            searchKey = gap
            if (self.absFlag):
                searchKey = np.abs(searchKey)

            percentile = self.orderedGaps[stock].bisect_left(searchKey)
            currSize = len(self.gapQueue[stock])
            # To avoid having percentile as 1.0, since percentile <= percSize + 1
            percentile = percentile / (currSize + 2.0)
            perc[stock] = percentile

        return perc

    def updatePercentile(self, gapSize):
        # Update the values in the percentile objects

        if (self.currSamples >= self.winSize):
            # Updating the queue and removing elements from the tree
            for stock in self.stockList:
                lastVal = self.gapQueue[stock].popleft()
                self.orderedGaps[stock].remove(lastVal)
            self.currSamples -= 1

        for stock, gap in gapSize.iteritems():
            searchKey = gap
            if (self.absFlag):
                searchKey = np.abs(searchKey)
            self.gapQueue[stock].append(searchKey)
            self.orderedGaps[stock].add(searchKey)
        self.currSamples += 1

    def getVolAvgPrice(self, opens, close, vol, left, right):
        '''
        Computes the volume weighted price for the range [left, right)
        price = (open + close)/2
        '''

        avgPrice = (opens.iloc[left:right] + close.iloc[left:right])/2.0
        volAvgPrice = (avgPrice * vol[left:right]).sum() / (vol[left:right].sum() + eps)

        return volAvgPrice

    def generateSignal(self, backData, currPos):
        # Generate signal should only be called when the day begins i.e. after a minute

        currTime = backData['open'].iloc[currPos].name
        currTimeStamp = datetime.fromtimestamp(currTime)
        currDay  = currTimeStamp.date()
        currHour = currTimeStamp.time().hour
        currMins = currTimeStamp.time().minute

        order = pd.DataFrame(0, index = backData['open'].columns, columns = ['signal', 'qty', 'position'])
        if (self.stopLoss):
            order['stopLoss'] = -1.0

        if (not ((currHour == 9) and (currMins == 15 + self.window))):
            return order

        opens = backData['open'][:currPos]
        close = backData['close'][:currPos]
        vol   = backData['vol'][:currPos]

        currOpen = self.getVolAvgPrice(opens, close, vol, currPos - self.window, currPos)
        prevClose = self.getVolAvgPrice(opens, close, vol, currPos - (2 * self.window), currPos - self.window)
        gapSize = (currOpen - prevClose) / (prevClose + eps)

        alpha = -gapSize
    
        # Percentile based filtering
        if (self.percFlag):
            self.updatePercentile(gapSize)
            if (self.currSamples >= self.winSize):
                perc = self.getPercentile(gapSize)
            else:
                return order
    
            alpha[perc < self.threshold] = 0.0

        volN = 70

        if (self.hedge == 'volBased'):
            if (opens.shape[0] < volN):
                return order

        vol = self.getRollVol(opens, period = volN)
        gapSize = gapSize / (vol + eps)

        beta = ((gapSize * 0.0) + 1.0)
        if (self.hedge == 'volBased'):
            beta = vol / vol[self.index]

        alphaNorm = np.sign(alpha)
        numIndex = -np.sum(alphaNorm * beta)
        alphaNorm[self.index] += numIndex
        # Normalizing the positions
        alphaNorm = alphaNorm / (np.abs(alphaNorm).sum() + eps)

        if (debug):
            print 'Normalized Alpha:', alphaNorm

        order['signal'] = np.sign(alphaNorm)
        order['qty'] = np.abs(alphaNorm)
        order['position'] = alphaNorm

        if (self.stopLoss):
            order['stopLoss'] = self.stopVal

        return order

class UnifiedSample(Strategy):

    def __init__(self, config):
        self.window = config['WINDOW']
        self.index  = config['INDEX']
        self.hedge  = config['HEDGE_METHOD']
        self.hold   = config['HOLD']
        self.percFlag = config['PERC_FLAG']
        self.limitFlag = config['LIMIT_FLAG']
        self.stopVal   = config['STOP_LOSS_VAL']
        self.stockList = config['STOCK_LIST']
        self.hedgeFlag = config['HEDGE']
        self.window    = config['VWAP_GAP_WINDOW']

        self.analyser = Analyser(config)

    def analyseTrades(self, dataDf):

        self.analyser.setPriceData(dataDf)
        # tradeList = self.analyser.getGapStats([self.hold])
        tradeList = self.analyser.getGapStats(range(2, self.hold + 3, 1))
        # tradeList = self.analyser.getGapStats(range(2, 361, 3))
        # tradeList = self.analyser.getGapStats([3, 6, 15, 30, 60, 90, 120, 180, 240, 300, 360])
        # tradeList = self.analyser.getGapStats(range(3, 30, 3) + range(30, 120, 6) + range(120, 361, 12))
        # self.analyser.printStats([self.hold])
        # self.analyser.printStats([3, 30, 60, 90, 180, 360])

        self.timeSet = {}
        for stock in self.stockList:
            self.timeSet[stock] = set(self.analyser.tradeList[stock].index)

        return tradeList

    def globalPrint(self, globalTradeList):
        # self.analyser.printStats([self.hold], globalTradeList, globalFlag = True)
        # self.analyser.printStats((range(3, 30, 3) + range(30, 120, 6) + range(120, 301, 12)), globalTradeList, globalFlag = True)
        # self.analyser.printStats([3, 15, 30, 60, 90, 120, 180, 240, 300, 360], globalTradeList, globalFlag = True)
        self.analyser.printStats([30, 60], globalTradeList, globalFlag = True)
        # self.analyser.printStats([self.hold], globalTradeList, globalFlag = True)

    def tradeCriteria(self, trade):
        '''
        Effectively represents the strategy in the new framework
        Passes a potential trade vector to check if it's valid
        trade contains the following columns,
        'currOpen', 'prevClose', 'entryPrice', 'gapSize', 'dailyRet',
        'signal', 'vol', 'gapRatio', 'percentile', 'finClose_hold', ....
        Can be made of (In the current scenario),
            - Filters based on gapSize, Percentile, etc
        '''

        votes = 0
        if (trade['openInLowHigh'] < -0.0118):
            votes += 1
        if ((1.0 - trade['vol']) < 0.985):
            votes += 1

        if (votes >= 2):
            return trade['signal']
        else:
            return 0
        # if (trade['percentile'] >= 90.0):
        #     return trade['signal']
        # if (trade['round_pcile'] >= 0):
        #     return trade['signal']
        # if (trade['profit_' + str(self.hold)] < 0.0001):
        #     return trade['signal']
        # else:
        #     return 0

    # @profile
    def generateSignal(self, currTime):

        order = pd.DataFrame(0, index = self.stockList, columns = ['signal', 'qty', 'position'])
        alpha = order['signal'] * 0.0
        beta  = order['signal'] * 0.0
        hold  = (order['signal'] * 0.0) + self.hold

        for stock in self.stockList:
            absent = ((currTime - (self.window * 60)) not in self.timeSet[stock])
            if absent:
                alpha[stock] = 0.0
            else:
                trade = self.analyser.tradeList[stock].loc[currTime - (self.window * 60)]
                alpha[stock] = self.tradeCriteria(trade)
                if (self.hedgeFlag):
                    beta[stock]  = trade['beta']
                if (self.limitFlag):
                    hold[stock] = trade['limit_hold_' + str(self.hold)]

        alphaNorm = np.sign(alpha)

        if (self.hedgeFlag):
            numIndex = -np.sum(alphaNorm * beta)
            alphaNorm[self.index] += numIndex
            # Required in holding period computation in execStrat
            order['beta'] = beta

        # Normalizing the positions
        # alphaNorm = alphaNorm / (np.abs(alphaNorm).sum() + eps)

        if (debug):
            print 'Normalized Alpha:', alphaNorm

        order['signal'] = np.sign(alphaNorm)
        order['qty'] = np.abs(alphaNorm)
        order['position'] = alphaNorm
        order['hold'] = hold

        return order
