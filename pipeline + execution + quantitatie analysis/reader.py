import pandas as pd
import numpy as np
import os
from dataStore import *

debug = False

def getNumLines(fileName):
    return sum(1 for line in open(fileName))

def getStockList(filesPath, year, numStocks):
    '''
    Get a random list of stocks for which the pipeline needs to
    executed for. For now only select stocks with complete info
    '''

    # filespath must be terminated by a /
    if (filesPath[-1] != '/'):
        filesPath += '/'

    fileList = os.listdir(filesPath + str(year))
    stockList = []
    cnt = 0
    while (len(stockList) < numStocks):
        numLines = getNumLines(filesPath + str(year) + '/' + fileList[cnt])
        if (numLines < 312):
            # Hard-coded for now. Refers to complete data for all days
            cnt += 1
            continue
        stockList.append(fileList[cnt])
        cnt += 1

    return stockList


class Reader():

    def __init__(self, config):
        '''
        Initialize the reader object
        Critical features include the file path
        The date ranges (Only year considered for now)
        The lookback depth (Given a day, how much can it peek back) (No Need)
        The stockList i.e. the stocks to take data for
        Consequently, we also have numStocks
        '''

        # fileDir must be terminated by a /
        if (config['PRICE_FILE_PATH'][-1] != '/'):
            config['PRICE_FILE_PATH'] += '/'

        self.year      = config['YEAR']
        self.fileDir   = config['PRICE_FILE_PATH'] + str(self.year) + '/'
        self.stockList = config['STOCK_LIST']
        self.numStocks = len(self.stockList)
        self.dataList  = None
        self.dataDf    = None
        self.dataStore = dataStore(config)

    def loadPriceData(self):
        '''
        Loads price data using dataStore interface
        '''
        self.dataStore.loadPriceData(minimal = True)

    def reshapeData(self, dataList):
        '''
        Assumes that all stocks have information at all common timestamps
        Originally dataList is of the form, {Stocks->{Dataframe(TimeStamp, Att.)}}
        The function modifies and reshapes it to {Att.->Dataframe(TimeStamp, Stocks)}
        '''

        newDataList = {}
        attList = dataList[self.stockList[0]].columns.values

        for att in attList:
            dfList = []
            for stock in self.stockList:
                dfList.append(dataList[stock][att])
            newDataList[att] = pd.concat(dfList, axis = 1)
            newDataList[att].columns = self.stockList

        self.dataList = newDataList
        self.attList = ['open', 'close', 'low', 'high', 'vol']

    def loadData(self, verbose = True):
        '''
        LEGACY FUNCTION: No longer used
        Returns a dictionary containing data for each stock
        We refer to the following by data i.e.
        For each stock,
        the time, open, close, low, high of all days
        '''

        dataList = {}
        for stock in self.stockList:
            # The files are stored in tsv format, therefore use read_table
            if verbose:
                print 'Reading', stock
            # Use cols mention which columns to keep, the chosen columns refer to time and prices
            # index_col refers to indexing relative to subset of columns passed in usecols
            dataList[stock] = pd.read_table(self.fileDir + stock, index_col = 0, \
                                            usecols = [0, 5, 6, 7, 8, 9], names = ['time', 'open', 'close', 'low', 'high', 'vol'])
            # NOTE: Column mappings are hard coded
            print dataList[stock].shape

        print 'Reading Complete'

        self.reshapeData(dataList)
        #print(self.dataList)

    def getStockList(self):
        '''
        Returns the stock List
        '''
        return self.stockList

    def fillDataOptimized(self):
        """
        Optimized version of the above function using pandas functions
        Fills up the price data for each timestamp in the union of the timestamps
        Returns a dataList required from the reader class
        """

        allTimeIdx = self.dataStore.getUnionTimeStampsOptimized()
        # Fills missing data
        self.dataStore.fillMissingTimeData(allTimeIdx)
        dataList = self.dataStore.priceDataDf

        self.dataDf = dataList
        return self.dataList


if __name__ == "__main__":

    filesPath = '/home/anishshah/Desktop/data/Dataset/Cash_data/'

    # stockList = getStockList(filesPath, year, numStocks)
    # stockList = ['CESC','DABUR','DRREDDY']
    stockList = ['ICICIBANK', 'HDFCBANK', 'YESBANK', 'FEDERALBNK', 'SBIN', 'INDUSINDBK', 'CANBK', 'KOTAKBANK', 'IDFCBANK', 'PNB']
    numStocks = len(stockList)

    config = {}
    # config['PRICE_FILE_PATH']   = '/home/nishantrai/Documents/data/Training_Data/Cash_data/'
    config['TRADE_FILE_PATH']   = None
    config['PRICE_FILE_PATH'] = '/home/nishantrai/Documents/data/Training_Data/Adjusted_FUT_Data/'
    config['YEAR']       = 2016
    config['STOCK_LIST'] = stockList

    print 'Stock List is:', stockList

    read = Reader(config)
    filler = DataFiller(config)
    dataList = filler.fillData()

    # print dataList
    # print read.reshapeData(dataList)

    # read.loadData()
    print 'Loading data finished'
