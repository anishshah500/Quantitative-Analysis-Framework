from dataStore import *
from visualise import *
from metrics import *
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import os
import progressbar
from datetime import datetime
from bdateutil import isbday

np.set_printoptions(precision = 3, suppress = True)

debug = False

def dt2ut(dt):
    epoch = pd.to_datetime('1970-01-01')
    return (dt - epoch).total_seconds()

class backTester():

    def __init__(self, config):
        '''
        Initializes the back-tester with the params specified in config
        Start time and end time represent the trades which should be executed
        i.e. Only trades inside the range are executed
        '''

        # Initialize basic variables from config
        self.timeStart  = config['START_TIME']
        self.tradeFile  = config['TRADE_FILE_PATH']
        self.priceFile  = config['PRICE_FILE_PATH']
        self.initBudget = config['INIT_BUDGET']
        self.stockList  = config['STOCK_LIST']
        self.logPeriod  = config['LOG_PERIOD']
        self.startTime  = config['START_TIME']
        self.endTime    = config['END_TIME']
        self.append     = config['APPEND']
        # Flags for logging details
        self.logPos     = config['LOG_POSITION']
        self.commission = config['TRADE_COMMISSION']

        # TODO: Not implemented yet, assuming params and semantics for now
        # dataStore contains trade data, price data and updCurrPrice()
        self.dataStore  = dataStore(config)

        # Hash map from stock name to id
        self.stockId   = {}
        for i in range(len(self.stockList)):
            self.stockId[self.stockList[i]] = i

        self.tradeLog   = True
        # State variables
        self.numStocks  = len(self.stockList)
        self.currBudget = 0
        self.currVal    = 0
        # Converting dictionaries to numpy arrays
        self.currPos    = np.zeros(self.numStocks)
        self.currPnl    = np.zeros(self.numStocks)
        self.totPnl     = np.zeros(self.numStocks)
        # List storing the periodic logs
        self.statList   = []
        # Storing the trading costs
        self.tradeCost  = 0
        self.currTradeCost  = 0

        print 'Reading Trade Log...'
        self.dataStore.loadTradeData()
        print 'Finished reading Trade Log'
 
        print 'Reading Price Log...'
        self.dataStore.loadPriceData()
        print 'Finished reading Price Log'

        self.nextLogTime = config['START_TIME']
        self.time = config['START_TIME']
        # self.time = self.dataStore.time
        # self.nextLogTime = self.startTime

    def getCommission(self):
        # Can modify later
        return self.commission

    def updTime(self, inc):
        '''
        Increases the time by inc amount
        Changes both self.time and self.dataStore.time
        '''

        # Business day support
        prevDate = datetime.fromtimestamp(self.time).date()
        nextTimeHour = datetime.fromtimestamp(self.time + inc).time().hour
        nextDate = datetime.fromtimestamp(self.time + inc).date()
        secInDay = 86400

        if (debug):
            print 'Time before:', self.time, self.dataStore.time
            print 'Hour after increment:', nextTimeHour
            print 'Time right now:', datetime.fromtimestamp(self.time).time()

        if (not (nextTimeHour >= 9 and nextTimeHour < 16) or (not isbday(nextDate))):
            if ((nextTimeHour >= 16) or (not isbday(nextDate))):
                self.time = dt2ut(datetime.fromtimestamp(self.time) + (1) * BDay())
            self.time = dt2ut(datetime.fromtimestamp(self.time).replace(hour=3, minute=30, second=0, microsecond=0))
            self.dataStore.time = self.time
            self.nextLogTime = self.time
            if (debug):
                print (datetime.fromtimestamp(self.time).date(), datetime.fromtimestamp(self.time).time())
        else:
            self.time += inc
            self.dataStore.time += inc
            self.nextLogTime += inc
            if(debug):
                print (datetime.fromtimestamp(self.time).date(), datetime.fromtimestamp(self.time).time())
            
        if (debug):
            print 'Time After:', self.time, self.dataStore.time
            # raw_input('Time updated...')
        # print self.time
        # print pd.to_datetime(self.time, unit='s')

    def printStats(self):
        print '\n' + ''.join(['*']*50)
        print 'TradeLog flag is:', self.tradeLog
        print 'Current backtester time is:', self.time
        print 'Current dataStore time is:', self.dataStore.time
        print 'Current nextLog time is:', self.nextLogTime
        print 'currBudget:', self.currBudget
        print 'currVal:', self.currVal
        print 'Current trade cost:', self.currTradeCost
        print 'Running trade cost:', self.tradeCost
        print 'Stock List:', self.stockList
        print 'Current Open:', self.dataStore.currOpen
        print 'Current Close:', self.dataStore.currClose
        print 'currPnl:', self.currPnl
        print 'totPnl:', self.totPnl
        print 'currPos:', self.currPos
        print ''.join(['*']*50) + '\n'

    def logStats(self, statList, colList, saveName = 'backtestLog.csv'):
        '''
        Logs the results of the back test simulation in to a csv file
        Args:
            statList: List of tuples containing different statistics
            colList : List of column names to be used while saving
            savePath: File to which it should be saved
        Returns:
            None
        '''

        if not os.path.exists('results/'):
            os.makedirs('results/')

        df = pd.DataFrame(self.statList, columns = colList)

        if (self.append):
            df.to_csv('results/' + saveName, mode = 'a', header = False, index = False)
        else:
            df.to_csv('results/' + saveName, index = False)

    def periodicLog(self):

        if (debug):
            print '\n' + ''.join(['*']*50)
            print 'Current backtester time is:', self.time
            print 'Current dataStore time is:', self.dataStore.time
            print 'Current nextLog time is:', self.nextLogTime
            print "Current Opening:", self.dataStore.currOpen
            print "Current closing:", self.dataStore.currClose
            print "Previous closing:", self.dataStore.prevClose
            print "Previous closing:", self.dataStore.prevOpen
            print ''.join(['*']*50) + '\n'

        self.dataStore.updCurrPrices()
        self.refreshLog()

    def refreshLog(self):
        '''
        Whenever, it is invoked then the state is refreshed
        by computing statistics like PnL, Total Pnl, val, etc
        PnL and tradeCost are computed with respect to the last
        time refreshLog was called.
        '''

        newVal = self.currBudget + np.dot(self.currPos, self.dataStore.currClose)
        self.currPnl = newVal - self.currVal
        self.totPnl += self.currPnl
        self.currVal = newVal
        self.currTradeCost = self.tradeCost
        self.tradeCost = 0

        tmpStat = [self.time, self.currVal, self.currBudget, np.sum(self.currPnl), np.sum(self.totPnl), self.currTradeCost]
        # Append the positions to the appended stats
        if (self.logPos):
            tmpStat.extend(list(self.currPos))

        self.statList.append(tmpStat)

    def backTest(self, tradeLog = False, verbose = False):
        '''
        Performs backtesting based on the trade list and price data
        dataStore must be invoked and updated before calling this
        as it stores the required state to do the backtesting
        Simulates and logs the results
        Args:
            tradeLog: Whether we log only at logPeriod or each trade as well
            verbose : Whether to print detailed statistics
        Returns:
            None
        '''

        self.tradeLog   = tradeLog
        # Original amount of funds
        self.currBudget = self.initBudget
        self.currVal    = self.currBudget
        # Will store the trade dates and pnl from the trades
        self.statList   = []
        self.totPnl     = 0
        self.tradeCost  = 0
        numTrades       = len(self.dataStore.tradeDataList)

        if (debug):
            print 'Before Beginning:'
            print 'currPos:\n', self.currPos
            print 'currBudget:', self.currBudget

        maxIters = int(((self.endTime - self.startTime) / self.logPeriod) + 1)
        maxIters += numTrades

        print 'Number of max iters predicted:', maxIters
        print 'Date range:', self.startTime, self.endTime

        print 'Current backtesting progress:'
        bar = progressbar.ProgressBar(maxval = (maxIters), \
                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        progId = 0
        startId = 0

        bar.start()

        self.time = self.startTime
        self.dataStore.time = self.startTime
        self.nextLogTime = self.startTime

        # Update time adds business days, so we shouldn't use it here
        self.updTime(self.startTime - self.time)

        # tradeList is a list/dataframe of trades
        for cnt in range(startId, numTrades):

            trade = self.dataStore.tradeDataList[cnt]

            # Using lists instead of dataframes since indexing is much slower
            # Note that, 0->time, 1->stock, 2->signal, 3->qty
            (time, stock, signal, qty) = (trade[0], trade[1], trade[2], trade[3])

            if (not (self.startTime <= time <= self.endTime)):
                continue

            # Print current trade timestamp
            if (debug):
                print '\n' + ''.join(['*']*50)
                print 'Current timestamp is:', time
                print 'Current backtester is:', self.time
                print ''.join(['*']*50) + '\n'

            # Periodically keep computing values
            while (self.nextLogTime < time):
                progId += 1
                if (progId%1 == 0):
                    bar.update(progId)
                self.periodicLog()
                self.updTime(self.logPeriod)
                if (verbose):
                    self.printStats()

            # Reset it in case it goes ahead of the current trade time
            # Which will always be the case
            self.dataStore.time = time
            self.time = time
            # Get current price
            self.dataStore.updCurrPrices()

            if (debug):
                print 'Trade details for stock:', stock, signal, qty
                print "Current Opening:", self.dataStore.currOpen[self.stockId[stock]]
                print "Current closing:", self.dataStore.currClose[self.stockId[stock]]
                print "Previous closing:", self.dataStore.prevClose[self.stockId[stock]]

            # Get signed quantity
            qty = qty * signal
            # Perform the singular trade, update the state
            self.tradeCost += self.performSingleTrade(stock, qty)

            if (tradeLog):
                self.refreshLog()

            if (verbose):
                self.printStats()

            if (debug):
                raw_input('Trade complete...')

        # Periodically keep computing values
        while (self.nextLogTime <= self.endTime):
            progId += 1
            if (progId%1 == 0):
                bar.update(progId)
            self.periodicLog()
            self.updTime(self.logPeriod)

        bar.finish()

        # Log the stat list to a file
        colList = ['time', 'value', 'currBudget', 'currPnl', 'totPnl', 'tradeCost']
        if (self.logPos):
            colList.extend(self.stockList)
        self.logStats(self.statList, colList)

        print len(self.statList)

        # raw_input('Backtest log finished (Enter to continue):')

    def performSingleTrade(self, stock, qty, tradeCostFlag = True):
        '''
        Simulates a single trade and returns the end results
        Args:
            stock: Stock for which the trade is performed
            qty: Quantity of stocks to buy (Signed)
            currBudget: Budget (liquid) before the trade
            currPos   : Position of stocks before the trade
            currPrice : Price at which the trade is happening
            tradeCostFlag : Whether to include tradeCosts in the simulation
        Returns:
            Current trade cost
            Has side effects i.e. (Modifies the variables in place)
       '''

        self.currPos[self.stockId[stock]] += qty

        if tradeCostFlag:
            # Assuming constant slippage for now
            slippage = self.getCommission()
            origPrice = self.dataStore.currOpen[self.stockId[stock]]
            slipPrice = origPrice * (1.0 + (np.sign(qty) * slippage))
            currTradeCost = np.abs(qty * (slipPrice - origPrice))
 
        self.tradeVal = (qty * origPrice)
        self.currBudget = self.currBudget - self.tradeVal - currTradeCost

        if (debug):
            print 'Arguments passed are:'
            print 'stock:', stock
            print 'qty:', qty
            print 'Performing trade:', self.tradeVal, currTradeCost

        return currTradeCost

def origCaller():

    config = {}
    config['INIT_BUDGET']     = 1500000
    # config['PRICE_FILE_PATH'] = "/home/anishshah/Desktop/data/Dataset/Cash_data/"
    config['PRICE_FILE_PATH'] = "/home/nishantrai/Documents/data/Training_Data/Adjusted_FUT_Data/"
    # config['TRADE_FILE_PATH'] = "results/tradeLog.csv"
    # config['STOCK_LIST']      = ['CESC','DABUR','DRREDDY']
    config['TRADE_FILE_PATH'] = "testcases/sample3.csv"
    # config['TRADE_FILE_PATH'] = "results/tradeLog.csv"
    # config['STOCK_LIST']      = ['NIFTY']
    # config['STOCK_LIST']      = ['ICICIBANK', 'HDFCBANK', 'YESBANK', 'FEDERALBNK', 'SBIN', 'INDUSINDBK', 'CANBK', 'KOTAKBANK', 'IDFCBANK', 'PNB']
    # config['STOCK_LIST']      = ['CIPLA','JINDALSAW','BANSWRAS']
    config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA', 'FEDERALBNK', 'YESBANK', 'BANKINDIA', 'CANBK', 'HDFCBANK', 'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'PNB', 'SBIN', 'UNIONBANK']
    # config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA', 'FEDERALBNK', 'IDBI', 'ORIENTBANK', 'YESBANK', 'BANKINDIA', 'CANBK', 'HDFCBANK', 'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'PNB', 'SBIN', 'UNIONBANK']
    config['NUM_STOCKS']      = len(config['STOCK_LIST'])
    config['LOG_PERIOD']      = 3600                # In seconds
    config['YEAR']            = 2014
    config['START_TIME']      = dt2ut(pd.to_datetime(str(config['YEAR']) + "/01/01"))
    config['END_TIME']        = dt2ut(pd.to_datetime(str(config['YEAR']) + "/12/31"))
    config['APPEND']          = False
    config['TRADE_COMMISSION']= 0.0005
    config['LOG_POSITION']    = True
    # config['NUM_STOCKS']      = 5
    # config['TRADE_FILE_PATH'] = 'testcases/dummyTrade.csv'
    # config['PRICE_FILE_PATH'] = 'testcases/dummyPrice.csv'

    tester = backTester(config)

    print 'Init backtester finished'

    import time

    startTime = time.time()

    tester.backTest(tradeLog = False,  verbose = False)

    print 'Back testing finished'

    print 'Time taken:', time.time() - startTime


def configCaller(fileName):

    import json

    with open(fileName) as jsonFile:    
        config = json.load(jsonFile)

    import time

    startTime = time.time()

    periods = config['PERIOD_LIST']
    config['NUM_DAYS'] = 0

    for i in range(len(periods)):
        # Converting the strings to date time format
        period = map(pd.to_datetime, periods[i])

        config['YEAR']       = period[0].year
        config['START_TIME'] = dt2ut(period[0])
        config['END_TIME']   = dt2ut(period[1])
        config['NUM_DAYS']   += (datetime.fromtimestamp(config['END_TIME']) - datetime.fromtimestamp(config['START_TIME'])).days
        config['APPEND']     = bool(i)

        tester = backTester(config)
        print 'Init backtester finished'

        tester.backTest(tradeLog = False,  verbose = False)

        config['INIT_BUDGET'] = tester.currBudget

        if (debug):
            print 'Period:', period, 'done with budget', config['INIT_BUDGET']

    print 'Back testing finished'
    print 'Time taken:', time.time() - startTime

    print 'Starting analysis phase'
    print_NAV_graph('results/backtestLog.csv')
    get_standard_ratios('results/backtestLog.csv', config)
    print 'Analysis phase finished'

if __name__ == '__main__':

    fileName = 'config/configGapUnified.json'
    # fileName = 'config/config.json'
    configCaller(fileName)
    # origCaller()
