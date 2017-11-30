import pandas as pd
import numpy as np
import os
import sys
import bisect
import progressbar
import multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Manager
import time

np.set_printoptions(precision = 3, suppress = True)

debug = False

def dt2ut(dt):
    epoch = pd.to_datetime('1970-01-01')
    return (dt - epoch).total_seconds()

class dataStore():

    def __init__(self, config):
        """
        Initializes the data-storage with the specified arguments
        """

        self.stockList  = config['STOCK_LIST']
        self.filedir_stock = config['PRICE_FILE_PATH']
        self.filedir_trades = config['TRADE_FILE_PATH']
        self.year = config['YEAR']

        self.time = 0
        # Setting to inf, to compute min easily
        self.minTime = sys.maxint
        # Setting to 0, to compute max easily
        self.maxTime = 0
        self.priceDataDf = {}
        self.priceDataList = {}
        self.priceTimeData = {}

        self.tradeDataList = []

        # Previous time statistics
        self.prevClose = np.zeros(len(self.stockList ))
        self.prevOpen  = np.zeros(len(self.stockList ))
        self.prevLow   = np.zeros(len(self.stockList ))
        self.prevHigh  = np.zeros(len(self.stockList ))
        # Current time statistics
        self.currClose = np.zeros(len(self.stockList ))
        self.currOpen  = np.zeros(len(self.stockList ))
        self.currLow   = np.zeros(len(self.stockList ))
        self.currHigh  = np.zeros(len(self.stockList ))
        self.currVol   = np.zeros(len(self.stockList ))

    def processMinimalPriceFiles(self, fileList):

        resultList = []

        for cnt in range(len(fileList)):

            stock = fileList[cnt]
            fileName = self.filedir_stock + str(self.year) + "/" + stock

            colList =['time', 'open', 'close', 'low', 'high', 'vol']
            priceDataDf = pd.read_csv(fileName, delimiter = "\t", index_col = 0,\
                                                  usecols = [0, 5, 6, 7, 8, 9], names = colList, header = None)
            resultList.append([stock, priceDataDf])

        return resultList

    def processPriceFiles(self, fileList):

        resultList = []

        for cnt in range(len(fileList)):

            stock = fileList[cnt]
            fileName = self.filedir_stock + str(self.year) + "/" + stock

            colList =['time', 'open', 'close', 'low', 'high', 'vol']
            priceDataDf = pd.read_csv(fileName, delimiter = "\t", index_col = 0,\
                                                  usecols = [0, 5, 6, 7, 8, 9], names = colList, header = None)
            # priceDataList = pd.read_csv(fileName, delimiter = "\t", header = None).values.tolist()
            priceDataList = pd.read_csv(fileName, delimiter = "\t", header = None).values.tolist()

            # Now changing to binary search based method to account for timestamps not in list
            priceTimeData = [priceDataList[i][0] for i in range(len(priceDataList))]

            resultList.append([stock, priceDataDf, priceDataList, priceTimeData])

        return resultList

    def processPriceFile(self, stock):

        fileName = self.filedir_stock + str(self.year) + "/" + stock

        colList =['time', 'open', 'close', 'low', 'high', 'vol']
        priceDataDf = pd.read_csv(fileName, delimiter = "\t", index_col = 0,\
                                              usecols = [0, 5, 6, 7, 8, 9], names = colList, header = None)
        priceDataList = pd.read_csv(fileName, delimiter = "\t", header = None).values.tolist()

        # Now changing to binary search based method to account for timestamps not in list
        priceTimeData = [priceDataList[i][0] for i in range(len(self.priceDataList))]

        return [priceDataDf, priceDataList, priceTimeData]

    def loadPriceData(self, minimal = False, verbose = True):
        """
        Stores the stock prices of all the stocks in the current stock list in 3 formats - pandas Frame, 
        List of Lists for faster indexing and dictionary of dictionaries
        Args:
            minimal: Only loads priceDataDf, much faster
            verbose: Prints multiple stats
        """

        startTime = time.time()

        NUM_PROCESSES = 20
        blockLen = int((len(self.stockList) * 1.0) / NUM_PROCESSES) + 1
        pool = ThreadPool(NUM_PROCESSES)

        processFunc = self.processMinimalPriceFiles if minimal else self.processPriceFiles
        resultList = pool.map(processFunc, [self.stockList[i*blockLen : (i+1)*blockLen] for i in range(NUM_PROCESSES)])

        pool.close()
        pool.join()

        for i in range(len(resultList)):
            for j in range(len(resultList[i])):
                stock = resultList[i][j][0]
                self.priceDataDf[stock] = resultList[i][j][1]
                if (not minimal):
                    self.priceDataList[stock] = resultList[i][j][2]
                    self.priceTimeData[stock] = resultList[i][j][3]

        print 'Time taken for reading data:', time.time() - startTime

    def loadTradeData(self):
        """
        Retrieves the data for all the stock orders and stores it as pandas frame and list of lists
        """

        signalMap = {'BUY': 1, 'HOLD': 0, 'SELL': -1, 0: 0, 1: 1, -1: -1}
        self.tradeDataList = pd.read_csv(self.filedir_trades)

        self.tradeDataList['signal'] = self.tradeDataList['signal'].map(signalMap)
        self.tradeDataList = self.tradeDataList.values.tolist()

    def updCurrPrices(self):

        time = self.time

        for i in range(len(self.stockList )):

            stock = self.stockList[i]
            rowPos = bisect.bisect(self.priceTimeData[stock], time) - 1

            # Since there is no entry related to this
            if (rowPos < 0):
                continue

            self.currOpen[i]  = self.priceDataList[stock][rowPos][5]
            self.currClose[i] = self.priceDataList[stock][rowPos][6]
            self.currLow[i]   = self.priceDataList[stock][rowPos][7]
            self.currHigh[i]  = self.priceDataList[stock][rowPos][8]
            self.currVol[i]   = self.priceDataList[stock][rowPos][9]

            # If this is the first time stamp, then prevOpen, close are same as the current ones
            if (rowPos == 0):
                rowPos += 1

            self.prevOpen[i]  = self.priceDataList[stock][rowPos - 1][5]
            self.prevClose[i] = self.priceDataList[stock][rowPos - 1][6]
            # Previos high] and low not needed
            # self.prevLow[i]   = self.priceDataList[stock][rowPos - 1][7]
            # self.prevHigh[i]  = self.priceDataList[stock][rowPos - 1][8]

            if (debug):
                i, stock
                print (self.prevOpen[i], self.prevClose[i], self.currOpen[i], self.currClose[i])            

    def getCurrPrices(self):
        '''
        Returns a dictionary of current prices
        The key is the stock name and the value
        is a list containing [open, close, low, high]
        No side effects
        '''

        currPrice = {}
        for i in range(len(self.stockList )):
            stock = self.stockList [i]
            currPrice[stock] = [self.currOpen[i], self.currClose[i], self.currLow[i], self.currHigh[i], self.currVol[i]]
        return currPrice

    def getUnionTimeStamps(self):
        '''
        Returns union of all time stamps in price file
        '''

        allTimeList = []
        for stock in self.stockList :
            allTimeList.extend(self.priceTimeData[stock])
       
        allTimeList = list(set(allTimeList))
        allTimeList.sort()

        return allTimeList

    def getUnionTimeStampsOptimized(self):
        '''
        Slightly optimized version of the above function using pandas functions
        '''
        idx = self.priceDataDf[self.stockList[0]].index
        for i in range(1, len(self.stockList)):                                                                      
            idx = idx.union(self.priceDataDf[self.stockList[i]].index)
        idx = idx[~idx.duplicated(keep = 'last')]
        return idx

    def fillMissingTimeData(self, idx, verbose = True):
        '''
        Fills the missing time data given in idx index
        Pads values using forward propagation
        '''

        if verbose:
            print 'Current processing progress:'
            bar = progressbar.ProgressBar(maxval = len(self.stockList), \
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

        for i in range(len(self.stockList)):                
            # Handle duplicate time indices
            self.priceDataDf[self.stockList[i]] = self.priceDataDf[self.stockList[i]][~self.priceDataDf[self.stockList[i]].index.duplicated(keep = 'last')] 
            # Fill in missing data for all time indices
            self.priceDataDf[self.stockList[i]] = self.priceDataDf[self.stockList[i]].reindex(idx)
            self.priceDataDf[self.stockList[i]].fillna(method = 'pad', inplace = True)
            # self.priceDataDf[self.stockList[i]].fillna(value = 0, inplace = True)
        
            if verbose:
                bar.update(i)

        if verbose:
            bar.finish()

if __name__ == '__main__':

    config = {}

    config['PRICE_FILE_PATH']   = '/home/nishantrai/Documents/data/Training_Data/Adjusted_FUT_Data/'
    config['TRADE_FILE_PATH']   = None
    # config['PRICE_FILE_PATH'] = "/home/nishantrai/Documents/data/Training_Data/Cash_data/"
    # config['TRADE_FILE_PATH'] = "results/tradeLog.csv"
    # config['STOCK_LIST'] = ['CESC','DABUR','DRREDDY']
    # config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA', 'NIFTY']
    # config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA', 'BANKINDIA', 'CANBK', 'HDFCBANK', 'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'PNB', 'SBIN', 'UNIONBANK']
    # NIFTY50: Works in 2010, 2012, 2014, 2016
    config['STOCK_LIST']      = ['ACC', 'AMBUJACEM', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BANKBARODA', 'BHARTIARTL', 'BHEL', 'BPCL',\
                                 'CIPLA', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDALCO',\
                                 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'ITC', 'KOTAKBANK', 'LT', 'LUPIN', 'M&M', 'MARUTI',\
                                 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMTRDVR', 'TATAPOWER', 'TATASTEEL',\
                                 'TCS', 'TECHM', 'ULTRACEMCO', 'WIPRO', 'ZEEL', 'CAIRN', 'PNB', 'NMDC', 'IDFC', 'DLF', 'JINDALSTEL']

    config['YEAR'] = 2010

    dataStore = dataStore(config)
    dataStore.loadPriceData(minimal = True)
    # dataStore.loadTradeData()
    # dataStore.updCurrPrices()