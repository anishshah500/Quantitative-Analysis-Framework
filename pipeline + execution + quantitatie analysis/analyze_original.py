from dataStore import *
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
from sortedcontainers import SortedList
from operator import truediv
from scipy.stats import ttest_ind
from bisect import bisect_left, bisect
import collections
from pandas import DataFrame
from scipy.stats import percentileofscore
import pickle

from pandas import pivot_table

eps = 1e-10
minInDay = 375

np.set_printoptions(precision = 2, suppress = True)

bucketTradeListGlobal = {}
for i in range(0,5):
    bucketTradeListGlobal[i] = {}

maxGapSize = 0

debug = False

grandDict = {}

class Analyse():

    def __init__(self, config):

        self.year      = config['YEAR']
        self.startTime = config['START_TIME']
        self.endTime   = config['END_TIME']
        self.stockList = config['STOCK_LIST']
        self.mode      = config['MODE']
        self.logBucket = config['LOG_BUCKET_DATA']
        self.hedgeFlag  = config['HEDGE']
        self.hedgeStock = config['HEDGE_STOCK']

        self.divideByVol = config['DIVIDE_BY_VOLATILITY']

        self.modStockList = self.stockList

        if (self.hedgeFlag):
            self.betaCorrelation = config['BETA_CORR']
            self.modStockList = [config['HEDGE_STOCK']] + self.stockList
            self.corrFlag = config['BETA_CORR_TYPE']

        if (self.mode == 'bucket'):
            self.bucketSize = config['BUCKET_SIZE']
            self.numBucket  = config['NUM_BUCKET']
        elif (self.mode == 'percentile'):
            self.bucketSize = config['BUCKET_SIZE']
            self.minSize    = config['MIN_SIZE']
            self.maxSize    = config['MAX_SIZE']
            self.absFlag    = config['ABS_FLAG']

        config['STOCK_LIST'] = self.modStockList
        # Datastore contains functions to read and update prices
        self.dataStore = dataStore(config)
        config['STOCK_LIST'] = self.stockList

        # Class members containing relevant Statistics
        # self.results: Dictionary containing stock names as keys
        #               Maps to a list of lists, where each list member
        #               contains gapSize, timeStamp, Open/Close prices
        #               along with holding periods, etc
        self.results            = {}
        self.gapListNormalized  = []

        self.prevCloseVWAPWindow    = config['VWAP_PREV_CLOSE_WINDOW']
        self.currOpenVWAPWindow     = config['VWAP_CURR_OPEN_WINDOW']
        self.posEntryVWAPWindow     = config['VWAP_POSITION_ENTRY_WINDOW']
        self.posExitVWAPWindow      = config['VWAP_POSITION_EXIT_WINDOW']

        self.printFlag = 0

        self.stopLoss = config['STOP_LOSS']
        self.targetPrice = config['TARGET_PRICE']

        self.tTestFlag = config['T_TEST_FLAG']
        if(self.tTestFlag):
            self.profitByGapPercentile = {}

            for i in range(0,100):
                self.profitByGapPercentile[i]  = []

        self.stockReturns = {}
        

    def loadData(self):
        '''
        Loads price data for the specified year and stock list
        Returns:
            None, only class members are modified
        '''
        self.dataStore.loadPriceData()

        for stock in self.stockList:
            price = pd.DataFrame(self.dataStore.priceDataList[stock][:]).iloc[:, 6]
            returns = ((price / price.shift(1)) - 1)[1:]
            self.stockReturns[stock] = returns
        if(self.hedgeFlag):
            price = pd.DataFrame(self.dataStore.priceDataList[self.hedgeStock][:]).iloc[:, 6]
            returns = ((price / price.shift(1)) - 1)[1:]
            self.stockReturns[self.hedgeStock] = returns
        # print(self.stockReturns)

    def getRetList(self, stock):

        price = pd.DataFrame(self.dataStore.priceDataList[stock][::minInDay]).iloc[:, 6]
        price = ((price / price.shift(1)) - 1)[1:]
    
        return price

    def getBenchmarkVolatility(self):

        price = pd.DataFrame(self.hedgePriceList[self.hedgeStock][::minInDay]).iloc[:, 6]
        price = ((price / price.shift(1)) - 1)[1:]
    
        return price

    def getVolatilityNDays(self, stock, n, currTimeRow):
        """
        Gets the volatility by taking returns of close prices for the last n days
        and does P(t) / P(t-1) - 1 for each of the n days and takes stDev
        """

        # price = pd.DataFrame(self.dataStore.priceDataList[stock][currTimeRow - 1 - (n * 375):currTimeRow - 1]).iloc[:, 6]
        # returns = ((price / price.shift(1)) - 1)[1:]

        returns = self.stockReturns[stock].iloc[currTimeRow - 1 - (n * 375):currTimeRow - 1]

        if(debug):
            print("Volatility: " + str(np.std(returns)))

        return np.std(returns)

    def getCorrelation(self, stock1, stock2, i1, i2, n):
        """
        Takes the prices of two stocks, calculates their return and gives their correlation
        """

        # price1 = pd.DataFrame(self.dataStore.priceDataList[stock1][-(n * 375) - 1 + i1:i1]).iloc[:, 6]
        # price2 = pd.DataFrame(self.dataStore.priceDataList[stock2][-(n * 375) - 1 + i2:i2]).iloc[:, 6]

        returns1 = self.stockReturns[stock1].iloc[i1 - 1 -(n * 375):i1 - 1]
        returns2 = self.stockReturns[stock2].iloc[i2 - 1 -(n * 375):i2 - 1]

        print(i1,i2)

        # print(returns1[-10:])
        # print(returns2[-10:])

        # if(len(price1) > len(price2)):
        #     # print("Price1: " + str(price1))
        #     # print("Price2: " + str(price2))
        #     price1 = price1[-len(price2):]
        #     print(i1,i2,len(price1),len(price2))

        # if(len(price2) > len(price1)):
        #     price2 = price2[-len(price1):]
        #     print(i1,i2,len(price1),len(price2))

        correlation = np.corrcoef(returns1,returns2)[1][0]

        return correlation


    def getVolAvgPrice(self, stock, left, right):
        '''
        Computes the volume weighted price for the range [left, right)
        price = (low + high + open + close)/4
        '''
        if(debug):
            print('\n'+ ''.join(['*']*50))
            print("Stock prices")
            print(left,right)
            print("Left price: " + str(self.dataStore.priceDataList[stock][left]))
            print("Right price: " + str(self.dataStore.priceDataList[stock][right]))

        price = np.array(self.dataStore.priceDataList[stock][left:right])[:, 5:]
        price = price.astype(np.float64)

        # 5, 6, 7, 8, 9: Open, Close, Low, High, Volume
        # After trimming off strings, 0, 1, 2, 3, 4: Opne, Close, Low, High, Volume
        avgPrice = (price[:,0] + price[:,1] + price[:,2] + price[:,3])/4.0
        volume = price[:,4]
        volAvgPrice = np.average(avgPrice, weights = volume)
        
        return volAvgPrice

    def getTTestScores(self, boundary, profitByGapPercentileLocal, verbose = False):
        #Returns the T test score and p-value of two arrays


        arr1 = []
        arr2 = []

        for i in range(1, boundary+1):
            arr1 += profitByGapPercentileLocal[i]

        for i in range(boundary+1, 99):
            arr2 += profitByGapPercentileLocal[i]
        
        tTest = ttest_ind(arr1,arr2)
        
        tValue, pValue = tTest[0], tTest[1]

        if(verbose):
            print("Boundary: " + str(boundary))
            print("T Value: " + str(tValue))
            print("P Value: " + str(pValue))

        return tValue,pValue

    def getGapStats(self, holdPeriodList, volType = 'nGapVol', verbose = False):
        '''
        Gives the statistics (Gap trading) for all hold periods specified
        The stats include
        timestamp, curr open price (after VWAP), prev close price (after VWAP), volatility
        holding period (H), min price/max price in interval, closing price after H etc
        Args:
            holdPeriodList: Contains holding periods as number of minutes
            volType; dailyVol or nDayVol (n = 30 by default)
        Returns:
            Dictionary as described above
        '''

        statList = {}
        priceList = {}
        gapList = {}

        if (self.hedgeFlag):
            # BM is benchmark
            gapListBM  = []
            volListBM  = []
            timeListBM = []
            # retList contains daily returns
            retListBM  = []
            priceListBM = []
            priceTimeBM = []
            #Stores all the timestamps for which the benchmark is indexed
            benchmarkTimeStamps = [eachList[0] for eachList in self.dataStore.priceDataList[self.hedgeStock]]

        volN = 70   # For standard volatility calculation of gapsize
        if (volType != 'stdVol'):
            volN = 30

        volDays = 70 # For standard volatility of entire calculation of returns
  
        for stock in self.modStockList:
            # Perform analysis for each stock

            infoList = self.dataStore.priceDataList[stock]
            statList[stock] = []
            priceList[stock] = []
            gapList[stock]  = []
            # gapListBenchmark[self.hedgeStock] = []

            retList = self.getRetList(stock)
            prevTime = 0
            print 'Currently analysing:', stock

            for i in range(len(infoList)):

                currTime = infoList[i][0]
                currTimeStamp = datetime.fromtimestamp(currTime)
                currDay  = currTimeStamp.date()
                currHour = currTimeStamp.time().hour
                currMins = currTimeStamp.time().minute

                # Account for duplicates
                if (prevTime == currTime):
                    continue
                prevTime = currTime

                if (not (self.startTime <= currTime <= self.endTime)):
                    # Check if it is in the valid range
                    continue

                if ((currHour == 9) and (currMins == 15)):
                    # Checking for day starting time

                    if(stock == 'SBIN' and currTimeStamp.date().day == 9 and currTimeStamp.date().month == 11 and self.year == 2016):
                        self.printFlag = 1

                    if(debug):
                        print('\n'+ ''.join(['*']*50))

                    #getting prices for stock
                    currOpen = self.getVolAvgPrice(stock, i, i+self.currOpenVWAPWindow)
                    prevClose = self.getVolAvgPrice(stock, i-self.prevCloseVWAPWindow, i)
                    posEntryPrice = self.getVolAvgPrice(stock, i + self.currOpenVWAPWindow, i + self.currOpenVWAPWindow + self.posEntryVWAPWindow)

                    if ((self.hedgeFlag) and (self.hedgeStock == stock)):
                        priceListBM.append(currOpen)
                        priceTimeBM.append(currTime)

                    priceList[stock].append(currOpen)
                    gapList[stock].append((currOpen - prevClose) / prevClose)

                    # Not enough samples to compute std dev, added five to handle edge cases
                    if (len(gapList[stock]) < volN + 5):
                        continue

                    # Refers to the stats common accross the holding periods
                    commStats = {}
                    commStats['time'] = currTime
                    commStats['readableTime'] = datetime.fromtimestamp(currTime)
                    commStats['ticker'] = stock
                    commStats['currOpen'] = currOpen
                    commStats['prevClose'] = prevClose
                    commStats['posEntryPrice'] = posEntryPrice

                    commStats['gapSize'] = ((currOpen - prevClose) / prevClose)

                    if(self.absFlag):
                        commStats['gapSize'] = np.abs(commStats['gapSize'])

                    if (volType == 'stdVol'):
                        commStats['volatility'] = np.std(retList[len(gapList[stock]) - volN: len(gapList[stock])])
                    else:
                        commStats['volatility'] = np.std(gapList[stock][-volN:])
                    commStats['gapRatio'] = commStats['gapSize'] / commStats['volatility']

                    #correct volatility using stDev of returns for 70 days of per minute returns
                    commStats['stockVolatility'] = self.getVolatilityNDays(stock,volDays,i)

                    if (self.hedgeFlag):
                        if (stock != self.hedgeStock):
                        
                            # Binary search in the timeStamps of the benchmark 
                            row = bisect(timeListBM, currTime) - 1
                            retBM = retListBM[row]
                            volBM = volListBM[row]

                            bmI = bisect_left(benchmarkTimeStamps, currTime)
                            posEntryBM = self.getVolAvgPrice(self.hedgeStock, bmI + self.currOpenVWAPWindow, bmI + self.currOpenVWAPWindow + self.posEntryVWAPWindow)

                            #modifying volatility
                            commStats['indexVolatility'] = self.getVolatilityNDays(self.hedgeStock,volDays,bmI)
                            
                            if(debug):
                                #Prints the timestamps of both the current stock row and the current benchmark row
                                print(self.dataStore.priceDataList[stock][i][0],self.dataStore.priceDataList[self.hedgeStock][bmI][0])

                            commStats['posEntryPriceBM'] = posEntryBM

                            if (self.corrFlag != 'constant'):
                                priceRow = bisect(priceTimeBM, currTime)
                                
                                # print len(priceList[stock][-volN:])
                                # print len(priceList[stock])
                                # print len(priceListBM)
                                # print -volN + priceRow, priceRow
                                # print len(priceList[self.hedgeStock][-volN + priceRow: priceRow])
                                self.betaCorrelation = np.corrcoef(priceList[stock][-volN:], priceListBM[-volN + priceRow: priceRow])[1][0]

                                # self.betaCorrelation = self.getCorrelation(stock,self.hedgeStock,i,bmI,volDays)

                            # beta = self.betaCorrelation * (volBM / commStats['volatility'])
                            # beta = self.betaCorrelation * (commStats['volatility'] / volBM)
                            beta = self.betaCorrelation * (commStats['stockVolatility'] / commStats['indexVolatility'])

                            if(debug):
                                print("Stock Volatility: " + str(commStats['stockVolatility']))
                                print("Index Volatility: " + str(commStats['indexVolatility']))

                            commStats['betaCorr'] = self.betaCorrelation
                            commStats['Beta'] = beta

                            if(verbose):
                                print(''.join(['*']*50))
                                print("Beta : " + str(beta))
                                print("Stock : " + stock)
                                print("Stock currOpen: " + str(currOpen))
                                print("Stock prevClose: " + str(prevClose))
                                print("Stock Return: " + str(commStats['gapSize']))
                                print("Stock Volatility: " + str(commStats['volatility']))
                                print("Stock Normalized Return: " + str(commStats['gapRatio']))
                                print("Benchmark Return: " + str(retBM))
                                print("Benchmark Volatility: " + str(volBM))
                                print("Benchmark Normalized Return: " + str(retBM / volBM))

                        else:
                            timeListBM.append(currTime)
                            retListBM.append(commStats['gapSize'])
                            volListBM.append(commStats['volatility'])

                    minPriceList = [float(infoList[i][6])]
                    maxPriceList = [float(infoList[i][6])]
                    # Identifying the array index limit
                    holdLim = min(max(holdPeriodList), len(infoList) - i - 1)
                    for j in range(holdLim):
                        minPriceList.append( min( minPriceList[-1], float(infoList[i + j][6])))
                        maxPriceList.append( max( maxPriceList[-1], float(infoList[i + j][6])))

                    #Appending volatility normalized gap value for determining distribution plot
                    self.gapListNormalized.append(commStats['gapSize'] / commStats['volatility'])

                    reachedStopOrTarget = 0
                    stopOrTargetRelReturn = 0

                    for hold in holdPeriodList:
                        tmpStats = commStats.copy()

                        minPrice = minPriceList[min(hold, holdLim)]
                        maxPrice = maxPriceList[min(hold, holdLim)]

                        tmpStats['holdPeriod'] = hold
                        tmpStats['min'] = minPrice
                        tmpStats['max'] = maxPrice
                        tmpStats['finClose'] = infoList[min((i + self.currOpenVWAPWindow + self.posEntryVWAPWindow + hold), len(infoList) -1)][6]

                        #Normalizing the volatility based on hold period
                        tmpStats['stockVolAfterNorm'] = commStats['stockVolatility'] * np.sqrt(hold)

                        if(self.hedgeFlag):
                            bmI = bisect_left(benchmarkTimeStamps, currTime)
                            tmpStats['finCloseBM'] = self.dataStore.priceDataList[self.hedgeStock][min((bmI + self.currOpenVWAPWindow + self.posEntryVWAPWindow + hold)\
                                , len(self.dataStore.priceDataList[self.hedgeStock]) - 1)][6]

                            # exitTime = infoList[i + hold][0]

                            # bmI = bisect_left(benchmarkTimeStamps, exitTime)
                            # tmpStats['finCloseBM'] = self.dataStore.priceDataList[self.hedgeStock][min((bmIExit), len(infoList) -1)][6]

                        if(not(stock == self.hedgeStock)):
                            #Calculating profits and all
                            tmpStats['profit'] = ((- np.sign(tmpStats['currOpen'] - tmpStats['prevClose'])) * \
                                ((tmpStats['finClose'] - tmpStats['posEntryPrice']) / tmpStats['posEntryPrice']))
                            tmpStats['absReturn'] = tmpStats['profit']
                            tmpStats['absReturnPerUnitVol'] = tmpStats['absReturn'] / tmpStats['stockVolAfterNorm']

                            if(self.hedgeFlag):
                                tmpStats['marketReturn'] = ((tmpStats['finCloseBM'] - tmpStats['posEntryPriceBM']) / tmpStats['posEntryPriceBM']) 
                                tmpStats['returnOffset'] = ((tmpStats['finCloseBM'] - tmpStats['posEntryPriceBM']) / tmpStats['posEntryPriceBM']) * tmpStats['Beta']
                                tmpStats['relReturn'] = tmpStats['profit'] + (np.sign(tmpStats['currOpen'] - tmpStats['prevClose']) * tmpStats['returnOffset'])
                                tmpStats['relReturnPerUnitVol'] = tmpStats['relReturn'] / tmpStats['stockVolAfterNorm']

                                if((tmpStats['relReturn'] <= self.stopLoss or tmpStats['relReturn'] >= self.targetPrice) and reachedStopOrTarget == 0):
                                    reachedStopOrTarget = 1
                                    stopOrTargetRelReturn = tmpStats['relReturn']

                                if(reachedStopOrTarget):
                                    tmpStats['relReturnWithStopLoss'] = stopOrTargetRelReturn
                                else:
                                    tmpStats['relReturnWithStopLoss'] = tmpStats['relReturn']

                            else:
                                tmpStats['relReturn'] = tmpStats['profit']
                                tmpStats['relReturnPerUnitVol'] = tmpStats['absReturnPerUnitVol']

                            # tmpStats['profitDividedByVol'] = tmpStats['relReturn'] / tmpStats['stockVolAfterNorm']


                        if(self.printFlag == 1):
                            for key in tmpStats:
                                print(key + ": " + str(tmpStats[key]))

                        statList[stock].append(tmpStats)

                        if(not(stock == self.hedgeStock)):
                            grandDict[stock].append(tmpStats)
                            # grandDF.append(tmpStats)

                    self.printFlag = 0

        self.results = statList

        # print sorted([statList[key][x]['gapRatio'] for key in statList.keys() for x in range(len(statList[key]))])[-1000:-900]

        return statList

    def compileResults(self, holdPeriodList):
        '''
        Compile the results extracted from getGapStats()
        The rows are indexed with RELATIVE RANK
        The columns are Count (For all stocks), also compute
        stock based results. E, P, R fraction.
        Win Rate: The fraction of actual fades
        Anti: Average profit on winning fade trades
        With: Average loss on losing fade trades 
        Exp: Expectation of profit
        Args:
            Hold period list, should be consistent with getGapStats()
        Returns:
            Matrix with the following column convention
            0: Count, 1: E, 2: P, 3: R, 4: P(S), 5: Anti, 6: With, 7:Exp
        '''

        self.timeWiseStats = {}
        self.cumStats = {}

        for hold in holdPeriodList:
            # numStocks rows, column mapping is given above
            if self.mode == 'relative':
                numRows = len(self.stockList)
            elif self.mode == 'percentile':
                numRows = int(100 / self.bucketSize)
            else:
                numRows = (2 * self.numBucket) + 1
            self.cumStats[hold] = np.zeros((numRows, 8))
            self.timeWiseStats[hold] = {}

            if (self.logBucket):
                # Stores a list of timetamps for each bucket
                # Stores a list of 
                self.bucketTimeList = []
                self.bucketTradeList = []
                tmpDict = {key : [] for key in self.stockList}
                for i in range(numRows):
                    self.bucketTimeList.append(list())
                    self.bucketTradeList.append(tmpDict.copy())

        for stockId in range(len(self.stockList)):

            stock = self.stockList[stockId]

            for i in range(len(self.results[stock])):

                tmpStats = self.results[stock][i]

                time      = tmpStats['time']
                hold      = tmpStats['holdPeriod']
                currOpen  = tmpStats['currOpen']
                prevClose = tmpStats['prevClose']
                minPrice  = tmpStats['min']
                maxPrice  = tmpStats['max']
                finClose  = tmpStats['finClose']
                gapRatio  = tmpStats['gapRatio']
                gapSize   = tmpStats['gapSize']
                # volatility= tmpStats['volatility']
                posEntry  = tmpStats['posEntryPrice']
                finClose  = tmpStats['finClose']

                volatility= tmpStats['stockVolAfterNorm']

                #Hedging support, not technically hedging, just offsetting with respect to the index return
                if(self.hedgeFlag):
                    hedge = ((tmpStats['finCloseBM'] - tmpStats['posEntryPriceBM']) / tmpStats['posEntryPriceBM']) * tmpStats['Beta']
                    if(self.divideByVol):
                        hedge /= volatility

                # Initial 8 elements represent the standard stats
                # The last ones will be used for ranking later
                tmpArr = np.zeros(12)
                tmpArr[0] += 1
                tmpArr[8] = stockId
                tmpArr[9] = gapSize
                tmpArr[10] = gapRatio
                tmpArr[11] = stockId
                
                targetPrice = finClose

                profit = ((- np.sign(currOpen - prevClose)) * ((targetPrice - posEntry) / posEntry))

                if(self.divideByVol):
                    profit /= volatility

                if (self.hedgeFlag):
                    profit -=  (-np.sign(currOpen - prevClose)) * hedge

                fillFlag = np.sign(profit)

                if (fillFlag < 0):
                    # Refers to the E case i.e. extension
                    tmpArr[1] += 1
                    tmpArr[6] += profit
                else:
                    if ((currOpen - prevClose) * (prevClose - targetPrice) < 0):
                        # Refers to the P case i.e. partial fill
                        tmpArr[2] += 1
                    else:
                        # Refers to the R case i.e. reversal
                        tmpArr[3] += 1
                    # Adding profits
                    tmpArr[5] += profit

                    
                # Adding the result to the corresponding time in the dict
                if (time not in self.timeWiseStats[hold]):
                    self.timeWiseStats[hold][time] = []
                self.timeWiseStats[hold][time].append(tmpArr)


        for hold in holdPeriodList:
    
            if self.mode == 'percentile':
                minSize = self.minSize
                maxSize = self.maxSize
                self.gapQueue = deque([], maxlen = maxSize)
                self.orderedGaps = SortedList(load = 50)

            for time in sorted(self.timeWiseStats[hold].keys()):

                if (self.mode == 'relative'):
                    # Sort the list according to the magnitude of gap size
                    self.timeWiseStats[hold][time].sort(key = lambda x: np.abs(x[-1]), reverse = True)
                    for i in range(len(self.timeWiseStats[hold][time])):
                        self.cumStats[hold][i] += self.timeWiseStats[hold][time][i][:8]

                elif (self.mode == 'percentile'):

                    newGapLen = len(self.timeWiseStats[hold][time])
                    newValList = []
                    # If there are enough elements for identifying percentile
                    if (len(self.gapQueue) >= minSize):

                        for i in range(newGapLen):
                            searchKey = self.timeWiseStats[hold][time][i][10]
                            # if (self.absFlag):
                            #     searchKey = np.abs(searchKey)
                            percentile = self.orderedGaps.bisect_left(searchKey)
                            currSize = len(self.gapQueue)
                            # To avoid having percentile as 1.0, since percentile <= percSize + 1
                            percentile = percentile / (currSize + 2.0)
                            row = int(percentile * int(100 / self.bucketSize))
                            # print row
                            self.cumStats[hold][row] += self.timeWiseStats[hold][time][i][:8]
                            if(self.tTestFlag):
                                self.profitByGapPercentile[int(percentile * 100)].append(self.timeWiseStats[hold][time][i][5] + self.timeWiseStats[hold][time][i][6])

                            if (self.logBucket):
                                # Adding time to this bucket's list
                                self.bucketTimeList[row].append(time)
                                # Since at least one of these is zero, by construction
                                profit = self.timeWiseStats[hold][time][i][5] + self.timeWiseStats[hold][time][i][6]
                                stockId = int(self.timeWiseStats[hold][time][i][11])
                                self.bucketTradeList[row][self.stockList[stockId]].append(profit)
                                bucketTradeListGlobal[row][self.stockList[stockId]].append(profit)

                        # Updating the queue and removing elements from the tree
                        for i in range(newGapLen):
                            lastVal = self.gapQueue.popleft()
                            self.orderedGaps.remove(lastVal)

                    for i in range(newGapLen):
                        searchKey = self.timeWiseStats[hold][time][i][10]
                        # if (self.absFlag):
                        #     searchKey = np.abs(searchKey)
                        newValList.append(searchKey)

                    # Adding the new values to the queue simultaneously
                    self.gapQueue.extend(newValList)
                    # Adding the new values to the tree simultaneously
                    self.orderedGaps.update(newValList)

                else:
                    for i in range(len(self.timeWiseStats[hold][time])):
                        # Sort the list according to the magnitude of gap size
                        gapRatio = self.timeWiseStats[hold][time][i][10]

                        # Get the position in the matrix, note that the bucket sizes are of size 10%
                        bucket = int(np.sign(gapRatio) * int(np.abs(gapRatio * 10) / self.bucketSize))
                        bucket = int(np.sign(bucket) * self.numBucket) if np.abs(bucket) >= self.numBucket else bucket
                        row = self.numBucket + bucket

                        self.cumStats[hold][row] += self.timeWiseStats[hold][time][i][:8]

    def tTestWrapper(self, profitByGapPercentile, verbose = True):
        """
        Tries various boundary values and gets the stats for each value from 1..99 as the boundary for percentile and
        Perfroms T Test on the profits >=value and <=value arrays
        """
        print(''.join(['*']*50))
        print("Cumulative Stats")


        if(self.tTestFlag):
            for i in range(10,100,10):
                tValue, pValue = self.getTTestScores(i,profitByGapPercentile)
                if(verbose):
                    print("Boundary: " + str(i))
                    print("T Value: " + str(tValue))
                    print("P Value: " + str(pValue))

    def getProfitGapPercentile(self):
        return self.profitByGapPercentile

    def finalizeStats(self, holdPeriodList):
        '''
        Finally processes the stats matrices, note that the resulting matrices
        cannot be compiled again directly as frequencies have become probs 
        '''

        for hold in holdPeriodList:
            self.cumStats[hold] = processStatMatrix(self.cumStats[hold])

    def plotDistribution(self, plotSeries, saveAsFile = False, logValues = False):
        '''
        Plots a histogram for the given plotsSeries
        Args:
            saveAsFile: Whether to save to file or plotting on screen
            logValues: Whether the y axis is log scaled
        Return:
            None, side effects could include saving a file
        '''

        stDev = np.std(plotSeries)

        #xLabels from -3*sigma to 3*sigma
        xLabels = np.array(range(-3,4)) * stDev

        plt.figure(figsize = (100,100))
        fig,ax = plt.subplots(1,1)
        axes = plt.gca()
        plt.hist(plotSeries,bins = 100,log = logValues)
        plt.xlabel("Normalized Gap Size")
        plt.ylabel("Number of Gap Sizes")
        axes.set_xlim([xLabels[0] - 0.5, xLabels[-1] + 0.5])

        ax.set_xticks(xLabels)

        plt.tight_layout()
        
        if(saveAsFile):
            plt.savefig("results/gapDistribution.svg")
        else:
            plt.show()

def processStatMatrix(data):
    '''
    Processes the passed data matrix i.e. fills in the Expectation,mean      0.000124
std       0.006253
    probability values, profit and losses
    '''
    # print(data[:,4])
    data[:, 4] = ((data[:, 2] + data[:, 3]) / (data[:, 0] + eps))
    data[:, 5] = (data[:, 5] / (data[:, 2] + data[:, 3] + eps))
    data[:, 6] = (data[:, 6] / (data[:, 1] + eps))
    data[:, 7] = (data[:, 5] * data[:, 4]) + (data[:, 6] * (1.0 - data[:, 4]))

    # Conversion to basis points
    # data[:, 4] *= 100.0
    # data[:, 5:] *= 10000.0

    data[:, :4] = np.round(data[:, :4], decimals = 0)
    # data[:, 4:] = np.round(data[:, 4:], decimals = 1)

    return data

def prettyPrintStats(stats, config):

    rowLab = []
    if (config['MODE'] == 'relative'):
        for i in range(len(config['STOCK_LIST'])):
            rowLab.append(str(i) + ':')
    elif (config['MODE'] == 'percentile'):
        for i in range(int(100 / config['BUCKET_SIZE'])):
            rowLab.append(str(i * config['BUCKET_SIZE']) + '%')
    else:
        for i in range(-config['NUM_BUCKET'], config['NUM_BUCKET'] + 1):
            rowLab.append(str(i * config['BUCKET_SIZE'] * 10) + '%')
    colLab = ['Count', 'E', 'P', 'R', 'P(S) (%)', 'Anti ', 'With ', 'Exp ']

    dfList = []
    for hold in sorted(stats.keys()):
        df = pd.DataFrame(stats[hold], columns=colLab, index=rowLab)
        dfList.append(df)

    dfList = pd.concat(dfList, axis=1, keys=sorted(stats.keys()))

    print 'Current Year:', config['YEAR']
    print dfList

    fileName = 'results/gapStats.csv'
    f = open(fileName, 'a')
    f.write('\nYear: ' + str(config['YEAR']) + '\n')
    f.close()
    print(dfList)
    dfList.to_csv(fileName, mode = 'a')

def getMultipleYearStats(config, yearList, holdPeriodList):

    dfList = []
    fullStats = {}
    # To store all the gaps throughput the year
    gapListNormalized = []
    profitByGapPercentile = {}

    fileName = 'results/gapStats.csv'
    f = open(fileName, 'w')
    f.close()

    for stock in config['STOCK_LIST']:
        for i in range(0,5):
            bucketTradeListGlobal[i][stock] = []

    for i in range(1,100):
        profitByGapPercentile[i] = []

    for hold in holdPeriodList:
        if (config['MODE'] == 'relative'):
            numRows = len(config['STOCK_LIST'])
        elif (config['MODE'] == 'percentile'):
            numRows = int(100 / config['BUCKET_SIZE'])
        else:
            numRows = (2 * config['NUM_BUCKET']) + 1

        fullStats[hold] = np.zeros((numRows, 8))

        if (config['LOG_BUCKET_DATA']):
            cumStdDict = None
            bucketTimeList = None

    for year in yearList:

        print 'Analysing year:', year

        config['YEAR']       = year
        config['START_TIME'] = dt2ut(pd.to_datetime(str(config['YEAR']) + "/01/01"))
        config['END_TIME']   = dt2ut(pd.to_datetime(str(config['YEAR']) + "/12/31"))

        analyser = Analyse(config)
        analyser.loadData()

        # analyser.getGapStats(holdPeriodList, volType = 'nGapVol')
        analyser.getGapStats(holdPeriodList, volType = 'stdVol')
        analyser.compileResults(holdPeriodList)
        gapListNormalized += analyser.gapListNormalized

        returnedGapList = analyser.getProfitGapPercentile()

        for i in range(1,100):
            profitByGapPercentile[i] += returnedGapList[i]
            # print(len(profitByGapPercentile[i]))

        for hold in holdPeriodList:
            fullStats[hold] += analyser.cumStats[hold]

        if (config['LOG_BUCKET_DATA']):
            if ((cumStdDict is None) or (cumTimeList is None)):
                cumStdDict = analyser.bucketTradeList
                cumTimeList = analyser.bucketTimeList
            else:
                for i in range(numRows):
                    cumTimeList[i].extend(analyser.bucketTimeList[i])
                    for stock in config['STOCK_LIST']:
                        cumStdDict[i][stock].extend(analyser.bucketTradeList[i][stock])

        analyser.finalizeStats(holdPeriodList)
        concatDf = prettyPrintStats(analyser.cumStats, config)
        
    for hold in holdPeriodList:
        fullStats[hold] = processStatMatrix(fullStats[hold])

    if (config['PLOT']):
        analyser.plotDistribution(gapListNormalized)
    
    print 'Statistics for normalized gaps:'
    print pd.Series(gapListNormalized).describe()
    analyser.tTestWrapper(profitByGapPercentile)

    config['YEAR'] = 'CUMULATIVE'
    prettyPrintStats(fullStats, config)

    # Printing important information such as unique timestamps and describe()
    if (config['LOG_BUCKET_DATA']):
        stdDevDict = {}
        for i in range(numRows):
            
            print 'For bucket', i
            print 'Original number of timestamps:', len(cumTimeList[i])
            print 'Unique number of timestamps:', len(set(cumTimeList[i]))
            countList = collections.Counter(cumTimeList[i])
            print 'Other statistics:', pd.Series(countList.values()).describe()

            stdDevDict[i] = {}
            for stock in config['STOCK_LIST']:
                stdDevDict[i][stock] = np.std(bucketTradeListGlobal[i][stock])

        print pd.DataFrame(stdDevDict)

if __name__ == "__main__":

    config = {}
    config['PRICE_FILE_PATH'] = "/home/anishshah/Desktop/data/Dataset/Adjusted_FUT_Data/"
    config['TRADE_FILE_PATH'] = None
    # BANKNIFTY: Works in all years
    # config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA', 'FEDERALBNK', 'IDBI', 'ORIENTBANK', 'YESBANK', 'BANKINDIA', 'CANBK',\
    #                              'HDFCBANK', 'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'PNB', 'SBIN', 'UNIONBANK']
    # config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA']
    # NIFTY50: Works in 2012, 2014
    # config['STOCK_LIST']      = ['ACC', 'AMBUJACEM', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BANKBARODA', 'BHARTIARTL', 'BHEL', 'BPCL',\
    #                              'CIPLA', 'COALINDIA', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HEROMOTOCO', 'HINDALCO',\
    #                              'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'INFY', 'ITC', 'KOTAKBANK', 'LT', 'LUPIN', 'M&M', 'MARUTI',\
    #                              'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMTRDVR', 'TATAPOWER', 'TATASTEEL',\
    #                              'TCS', 'TECHM', 'ULTRACEMCO', 'WIPRO', 'ZEEL', 'CAIRN', 'PNB', 'NMDC', 'IDFC', 'DLF', 'JINDALSTEL']
    # NIFTY50: Works in 2010, 2012, 2014, 2016
    # config['STOCK_LIST']      = ['ACC', 'AMBUJACEM', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BANKBARODA', 'BHARTIARTL', 'BHEL', 'BPCL',\
    #                              'CIPLA', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDALCO',\
    #                              'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'ITC', 'KOTAKBANK', 'LT', 'LUPIN', 'M&M', 'MARUTI',\
    #                              'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMTRDVR', 'TATAPOWER', 'TATASTEEL',\
    #                              'TCS', 'TECHM', 'ULTRACEMCO', 'WIPRO', 'ZEEL', 'CAIRN', 'PNB',  'IDFC', 'DLF', 'JINDALSTEL']
    # config['MODE']            = 'bucket'
    # config['BUCKET_SIZE']     = 3
    # config['NUM_BUCKET']      = 6
    # config['MODE']            = 'relative'
    config['MODE']            = 'percentile'
    config['ABS_FLAG']        = False
    config['BUCKET_SIZE']     = 20
    config['MIN_SIZE']        = 1500
    config['MAX_SIZE']        = 2000
    config['YEAR']            = 2012
    config['START_TIME']      = dt2ut(pd.to_datetime(str(config['YEAR']) + "/01/01"))
    config['END_TIME']        = dt2ut(pd.to_datetime(str(config['YEAR']) + "/12/31"))
    config['PLOT']            = False
    config['HEDGE']           = True
    config['HEDGE_STOCK']     = 'NIFTY'
    config['BETA_CORR_TYPE']  = 'constant'
    config['BETA_CORR']       = 1

    config['DIVIDE_BY_VOLATILITY'] = False

    config['T_TEST_FLAG']     = True

    # Useful utility flag for logging unique time stamps in each 
    # bucket, also for computing stddev, correlation, etc
    config['LOG_BUCKET_DATA'] = True

    #VWAP Window Parameters
    config['VWAP_PREV_CLOSE_WINDOW']        = 2
    config['VWAP_CURR_OPEN_WINDOW']         = 2
    config['VWAP_POSITION_ENTRY_WINDOW']    = 2
    config['VWAP_POSITION_EXIT_WINDOW']     = 2

    config['STOP_LOSS'] = -0.004
    config['TARGET_PRICE'] = 2

    # yearList = [2016]
    yearList =  [2010, 2012, 2014, 2016]
    # holdPeriodList = [4,8,12,20,28,40,60,80,100,120]
    # holdPeriodList = [30]
    # holdPeriodList = [30,60,120,240]
    # holdPeriodList = [15, 30]

    config['STOCK_LIST'] = [sys.argv[1]]

    holdPeriodList = []
    for i in range(1,361):
        holdPeriodList.append(i)

    for stock in config['STOCK_LIST']:
        grandDict[stock] = []

    getMultipleYearStats(config, yearList, holdPeriodList)

    # print(grandDict)

    # grandDf = pd.DataFrame(index = config['STOCK_LIST'])
    # for stock in config['STOCK_LIST']:
    #     grandDf[stock] = pd.DataFrame(grandDict[stock])

    # print(grandDf)
    # print(pd.Panel(grandDf))

    #initalizing the big dataframe and filling it with the stats of all years
    grandDf = dict()
    columns = grandDict[config['STOCK_LIST'][0]][0].keys()
    grandDF = pd.DataFrame()
    for stock in config['STOCK_LIST']:
        df = pd.DataFrame(grandDict[stock],columns = columns)
        df['gapSize'] = np.abs(df['gapSize'])
        df['gapRatio'] = np.abs(df['gapRatio'])
        df['percentile'] = df['gapRatio'][::360].rolling(window = 70, min_periods = 10).apply(lambda x: percentileofscore(x,x[-1]))
        df['percentile'] = df['percentile'].fillna(method = 'ffill')
        df['round_pcile'] = df['percentile'].dropna().apply(lambda x: int((x / 20.0) + 0.5) * 20)
        grandDf[stock] = df#.dropna()
        grandDF = grandDF.append(df)
        # print(grandDF)
    grandDf = pd.Panel(grandDf)
    # print(grandDf['ACC'])
    # print(grandDf['ACC'])

    # df1 = pd.DataFrame(grandDF,columns = columns)
    # df1['percentile'] = df1['gapSize'].rolling(window = 280).apply(lambda x: percentileofscore(x,x[-1]))
    # print(grandDF)
    # grandDF.to_csv('allTradesAnalyzeCSV.csv')

    # with open('allTradesPanel.pickle','wb') as handle:
    #     pickle.dump(grandDF, handle, protocol=pickle.HIGHEST_PROTOCOL)

    file_exists = os.path.isfile('allTradesAnalyzeNew.csv')

    if not file_exists:
        with open('allTradesAnalyzeNew.csv','a') as f:
            grandDF.to_csv(f, header = True)    
    
    with open('allTradesAnalyzeNew.csv','a') as f:
        grandDF.to_csv(f, header = False)

    print('\n')
    print("Version B")
    allPositiveTrades = grandDF[grandDF['relReturn'] >= 0]
    allNegativeTrades = grandDF[grandDF['relReturn'] < 0]
    print(''.join(['*'] * 50))
    print("All Positive Trades\n")
    print(pivot_table(allPositiveTrades, values = ['absReturn','absReturnPerUnitVol','relReturn','relReturnPerUnitVol'], index = ['round_pcile'], aggfunc = np.mean))
    print('\n')
    print(''.join(['*'] * 50))
    print("All Negative Trades\n")
    print(pivot_table(allNegativeTrades, values = ['absReturn','absReturnPerUnitVol','relReturn','relReturnPerUnitVol'], index = ['round_pcile'], aggfunc = np.mean))
    print('\n')
    print(''.join(['*'] * 50))
    print("All Trades\n")
    print(pivot_table(grandDF, values = ['absReturn', 'relReturn'], index = ['round_pcile'], aggfunc = [np.mean, np.std, np.max, np.min]))
    # print(pivot_table(grandDF, values = ['absReturn','absReturnPerUnitVol','relReturn','relReturnPerUnitVol'], index = ['round_pcile'], aggfunc = np.mean))
    print('\n')
    print(''.join(['*'] * 50))
    print("Gap Size, gap Ratio average\n")
    print(pivot_table(grandDF, values = ['gapSize','gapRatio'], index = ['round_pcile'], aggfunc = np.mean))
    print('\n')
    print("P(S)\n")
    # print(grandDF[grandDF['relReturn'] >= 0].iloc[:,1].groupby(grandDF['round_pcile']).count() / grandDF.iloc[:,1].groupby(grandDF['round_pcile']).count())
    print('\n')
    print(''.join(['*'] * 50))
    print(''.join(['*'] * 50))

    #Fact checking
    print("Type of correlation: " + str(config['BETA_CORR_TYPE']))
    print("Correlation: " + str(config['BETA_CORR']))
    print("Hedge: " + str(config['HEDGE']))
    print("Dividing by volatility: " + str(config['DIVIDE_BY_VOLATILITY']))

    # analyser = Analyse(config)
    # analyser.loadData()
    # analyser.getGapStats(holdPeriodList)
    # analyser.compileResults(holdPeriodList)
    # analyser.finalizeStats(holdPeriodList)
    # prettyPrintStats(analyser.cumStats)

