from strategy import *
from reader import *
import progressbar
import datetime
from collections import deque
from copy import deepcopy

np.set_printoptions(precision = 3, suppress = True)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 20)

debug = False
traceFlag = False

def dt2ut(dt):
    epoch = pd.to_datetime('1970-01-01')
    return (dt - epoch).total_seconds()

class SimStrategy():

    def __init__(self, config):
        '''
        Initialize the Simulate Strategy object
        Critical parts include a reader object
        and a strategy object, both of which
        have already been initialized
        '''

        self.cntTrades = 0

        self.reader   = Reader(config)
        # self.strategy = MovingAvg()
        # self.strategy = SimpleMom()
        # self.strategy = SimpleVol(config)
        # self.strategy = SimpleGapFade(config)
        self.strategy = UnifiedSample(config)

        # Reader specific functions
        self.reader.loadPriceData()
        self.reader.fillDataOptimized()
        self.reader.reshapeData(self.reader.dataDf)

        self.initBudget = config['INIT_BUDGET']
        self.currVal    = config['INIT_BUDGET']
        self.currBudget = config['INIT_BUDGET']
        self.startTime  = config['START_TIME']
        self.endTime    = config['END_TIME']
        self.flush      = config['FLUSH_AT_END']
        self.hedgeFlag  = config['HEDGE']
        self.hedgeStock = config['HEDGE_STOCK']

        # Refers to the amount of trading ticks
        self.hold       = config['HOLD']
        self.holdPeriod = config['HOLD_PERIOD']
        self.stopLoss   = config['STOP_LOSS_FLAG']
        self.limitFlag  = config['LIMIT_FLAG']
        self.fact       = config['TRADE_FACT']
        self.tradeFreq  = config['TRADE_FREQ']
        self.window     = config['VWAP_GAP_WINDOW']

        # dailyTrade is true if we only make trades in the beginning of the day
        self.dailyTrade = config['DAILY_TRADE']
        self.currQty    = None
        self.tradeCost  = 0

        if (self.holdPeriod > 0):
            self.holdBackList = deque([], maxlen = self.holdPeriod)
            if (self.stopLoss):
                self.stopLossList = deque([], maxlen = self.holdPeriod)

    def performAnalysis(self):
        tradeList = self.strategy.analyseTrades(self.reader.dataDf)
        return tradeList

    def globalPrintStrategy(self, globalTradeList):
        # print("GLOBAL : printFlg = True")
        self.strategy.globalPrint(globalTradeList)        

    def logOrders(self, logList, savePath = 'results/', append = False):
        '''
        Logs the computed trades to a file
        It receives a list of log records, each containing
        time, orders placed, current day prices
        Both orders and prices are dataframes with stocks as indexes
        '''

        if (traceFlag):
            print 'logOrders: START'

        print 'Started logging results'

        stockList = self.reader.stockList

        orderDataList = []
        priceDataList = []
        for (time, order, price) in logList:
            order = order.to_dict()
            price = price.to_dict()

            for stock in stockList:
                tmpOrder = [time, stock, order['signal'][stock], order['qty'][stock]]
                tmpPrice = [time, stock, price['open'][stock], price['close'][stock]]
                
                # Avoiding logging non traded stocks
                if (tmpOrder[3] != 0):
                    orderDataList.append(tmpOrder)
                    priceDataList.append(tmpPrice)

        # Creating the dataframe
        orderDf = pd.DataFrame(orderDataList, columns = ['time', 'stock', 'signal', 'qty'])
        priceDf = pd.DataFrame(priceDataList, columns = ['time', 'stock', 'open', 'close'])

        orderDf.to_csv(savePath + 'tradeLog.csv', index = False)
        priceDf.to_csv(savePath + 'priceLog.csv', index = False)

        print 'Logged order and price details'

        if (traceFlag):
            print 'logOrders: END'

    def processOrders(self, order, price, currPos):
        '''
        Given the desired positions convert to the 
        desired order format
        '''

        if (self.currBudget < 0):
            print 'WARNING: Budget is negative. Transactions may not take place'

        newPos = order['position']
        # Original method of computing order quantities
        # orderQty = ((newPos - currPos) / (price['close'] + eps)) * self.budget

        # currPos = (self.currQty * price['open']) / (np.abs(self.currQty * price['open']).sum() + eps)
        # newQty = ((newPos / (price['open'] + eps)) * self.currBudget * self.fact)
        # orderQty = newQty - self.currQty

        orderQty = ((newPos / (price['open'] + eps)) * self.currVal * self.fact)
        # orderQty = ((newPos / (price['open'] + eps)) * 100000.0 * self.fact)
        orderQty.fillna(0, inplace = True)

        if (self.holdPeriod):

            tmpHold = pd.DataFrame(0, index = orderQty.keys(), columns = ['qty', 'origPrice', 'stopLoss'])

            if (self.stopLoss):
                tmpHold['stopLoss'] = -1.0
                tmpHold['hold']     = 0.0

            if (self.hedgeFlag):
                tmpHold['beta'] = 0.0

            if (len(self.holdBackList) == 0):
                for i in range(self.holdPeriod):
                    self.holdBackList.append(deepcopy(tmpHold))

            tmpHold['qty'] = deepcopy(orderQty)
            tmpHold['origPrice'] = deepcopy(price['open'])

            holdBackOrder = self.holdBackList.popleft()['qty']
            
            if (self.stopLoss):
                tmpHold['stopLoss'] = order['stopLoss']
                tmpHold['hold']     = order['hold']
                if (self.hedgeFlag):
                    tmpHold['beta'] = order['beta']

                # Note that currently we have one less object in holdBackList
                for i in range(len(self.holdBackList)):
                    tmpOrder = self.holdBackList[i]
                    # qty, origPrice, stopLoss, *target
                    profit = np.sign(tmpOrder['qty']) * ((price['open']/(tmpOrder['origPrice'] + eps)) - 1)

                    if (self.hedgeFlag):
                        profitIndex = np.sign(tmpOrder['qty']) * tmpOrder['beta'] * \
                                      ((price['open'][self.hedgeStock]/(tmpOrder['origPrice'][self.hedgeStock] + eps)) - 1)
                        profit -= profitIndex

                    if (debug):
                        print 'Past Quantity, Profit, Price'
                        print pd.concat([tmpOrder['qty'], profit, tmpOrder['origPrice']], axis = 1)
                        print 'The stoplossed people are', profit[profit < tmpOrder['stopLoss']]

                        if (tmpOrder[profit < tmpOrder['stopLoss']].shape[0]):
                            print 'Hold Periods computed in Analysis:', tmpOrder[profit < tmpOrder['stopLoss']]['hold'].values
                            print 'Hold Period computed in execution:', (len(self.holdBackList) - i) * self.tradeFreq
                            print ''.join(['*']*20)

                    newOrder = profit * 0.0
                    # Stop loss not hit
                    # profit[profit >= tmpOrder['stopLoss']] = 0.0
                    newOrder[profit <= tmpOrder['stopLoss']] = deepcopy(tmpOrder['qty'])

                    # Only gets sold with other stocks in case of hedging
                    if (self.hedgeFlag):
                        newOrder[self.hedgeStock] = 0.0
                    # profit[np.abs(profit) > eps] = deepcopy(tmpOrder['qty'])

                    if (self.hedgeFlag):
                        numIndex = -np.sum((newOrder * tmpOrder['beta'] * tmpOrder['origPrice']) / tmpOrder['origPrice'][self.hedgeStock])
                        newOrder[self.hedgeStock] += numIndex
                        self.holdBackList[i]['qty'].loc[self.hedgeStock] -= numIndex

                    holdBackOrder += newOrder
                    # Updating the quantities as they have already been cleared
                    # self.holdBackList[i].loc[np.abs(profit) > eps, 'qty'] = 0
                    self.holdBackList[i].loc[profit <= tmpOrder['stopLoss'], 'qty'] = 0

            self.holdBackList.append(deepcopy(tmpHold))
            orderQty -= holdBackOrder

            if (debug):
                print 'Holdback order is', holdBackOrder
                # print 'Current holdBackList len:', len(self.holdBackList)
                # print 'Current holdBackList:', (self.holdBackList)

        if (debug):
            print 'Current order:', orderQty
            # print 'Prev qty:', self.currQty
            # print 'New Pos:', newPos
            print 'Check sum:', (orderQty * price['open']).sum()

        order['signal'] = np.sign(orderQty)
        # Avoid zero priced stocks since they are not trading currently
        order.loc[price['open'] < eps, 'signal'] = 0
        order['qty'] = np.abs(orderQty)
        order.loc[order['signal'] == 0, 'qty'] = 0

        # Since we perform integral orders, round to integer
        # order['qty'] = np.rint(order['qty'])
        order.loc[order['qty'] == 0, 'signal'] = 0

        return order

    def processOrdersOptimized(self, order, price, currPos):
        '''
        Alternative implementation of the process order function
        Given the desired positions convert to the desired order format
        NOTE: It assumes a difference method for executing holding period
        transactions i.e. it assumes that generate signals returns a 
        field which contains the desred holding period (including the effects
        of stop loss as well)
        Implemented to keep the stop loss computation consistent
        '''

        if (traceFlag):
            print 'processOrdersOptimized: START'

        if (self.currBudget < 0):
            print 'WARNING: Budget is negative. Transactions may not take place'

        newPos = order['position']
        # orderQty = ((newPos / (price['open'] + eps)) * self.currVal * self.fact)
        orderQty = ((newPos / (price['open'] + eps)) * self.currVal * 0.001)
        orderQty.fillna(0, inplace = True)

        if (self.holdPeriod):

            tmpHold = pd.DataFrame(0, index = orderQty.keys(), columns = ['qty'])

            if (len(self.holdBackList) == 0):
                for i in range(self.holdPeriod):
                    self.holdBackList.append(deepcopy(tmpHold))

            holdBackOrder = self.holdBackList.popleft()['qty']
            self.holdBackList.append(deepcopy(tmpHold))
            
            for stock in order[np.abs(orderQty) > eps].index:
                # We assume trade freq divides the holding period
                tmpHold = (int(order['hold'][stock]) / self.tradeFreq)

                if (stock != self.hedgeStock):
                    self.holdBackList[tmpHold - 1]['qty'].loc[stock] += orderQty[stock]
                    if (self.hedgeFlag):
                        tmpBeta = order['beta'][stock]
                        numIndex = ((tmpBeta * orderQty[stock] * price['open'][stock]) / price['open'][self.hedgeStock])
                        self.holdBackList[tmpHold - 1]['qty'].loc[self.hedgeStock] -= numIndex

            orderQty -= holdBackOrder

            if (debug):
                print 'Holdback order is', holdBackOrder
                # print 'Current holdback list is'
                # print pd.concat(self.holdBackList, axis = 1)

        if (debug):
            print 'Current order:', orderQty
            # print 'Prev qty:', self.currQty
            # print 'New Pos:', newPos
            print 'Check sum:', (orderQty * price['open']).sum()

        order['signal'] = np.sign(orderQty)
        # Avoid zero priced stocks since they are not trading currently
        order.loc[price['open'] < eps, 'signal'] = 0
        order['qty'] = np.abs(orderQty)
        order.loc[order['signal'] == 0, 'qty'] = 0

        # Since we perform integral orders, round to integer
        # order['qty'] = np.rint(order['qty'])
        order.loc[order['qty'] == 0, 'signal'] = 0

        if (traceFlag):
            print 'processOrdersOptimized: END'

        return order

    def runStrategy(self):

        if (debug):
            print 'Started computing trades'
    
        dataList = self.reader.dataList

        # Hardcoded due to the strategy used
        # Trading starts in January
        # startId = int(3 * 30 * jumpSize * 0.7)
        startId = 0
        endId   = dataList['open'].shape[0]
        # endId   = startId + 50
        currPos = 0 * dataList['open'].iloc[0]
        self.currQty = 0.0 * dataList['open'].iloc[0]
        self.currQty.fillna(0, inplace = True)
        logList = []

        print dataList['open'].shape, startId, endId

        print 'Current trading progress:'
        bar = progressbar.ProgressBar(maxval = endId, \
                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        tradeId = startId

        while (tradeId < endId):

            # Since the first column (and index) is time
            currTime = dataList['open'].iloc[tradeId].name

            if (not (self.startTime <= currTime <= self.endTime)):
                tradeId += self.tradeFreq
                continue

            currTimeStamp = datetime.datetime.fromtimestamp(currTime)
            currDay  = currTimeStamp.date()
            currHour = currTimeStamp.time().hour
            currMins = currTimeStamp.time().minute

            if (tradeId % 25000 == 0):
                print currTimeStamp
                # print 'Trade Cost:', self.tradeCost
                # print 'Current Qty:', np.array(self.currQty)
                print 'Current Budget:', self.currBudget
                print 'Current Val:', self.currVal
                print 'NumTrades:', self.cntTrades
                print 'Avg. Profit:', self.currVal, self.initBudget
                print 'Avg. Profit:', (((self.currVal - self.initBudget) / 1000.0) / (self.cntTrades + eps)) * 1e4

            if (self.dailyTrade):
                lowTime  = deepcopy(currTimeStamp).replace(hour = 9, minute = (15 + self.window))
                highTime = deepcopy(lowTime) + datetime.timedelta(minutes = (self.holdPeriod * self.tradeFreq))
                # print currTimeStamp, lowTime, highTime, (lowTime <= currTimeStamp <= highTime)
                if (not (lowTime <= currTimeStamp <= highTime)):
                    tradeId += 1
                    continue

            if (traceFlag):
                print 'generateSignal: START'
            currOrder = self.strategy.generateSignal(currTime)
            if (traceFlag):
                print 'generateSignal: END'

            # Saving the current day price
            currPrice = pd.DataFrame(0, index = dataList['open'].columns, columns = ['open', 'close'])
            currPrice['open'] = dataList['open'].iloc[tradeId]
            currPrice['close'] = dataList['close'].iloc[tradeId]

            self.cntTrades += (np.sum(np.abs(currOrder['qty']) > eps))

            if (debug):
                print currOrder

            # If we want to manipulate positions and other quantities
            # currOrder = self.processOrders(currOrder, currPrice, currPos)
            currOrder = self.processOrdersOptimized(currOrder, currPrice, currPos)
            currPos = currOrder['position']

            if (traceFlag):
                print 'loggingComputation: START'

            # Compute step statistics
            self.tradeCost = (currOrder['qty'] * currOrder['signal'] * currPrice['open']).sum()
            self.currQty += currOrder['qty'] * currOrder['signal']
            self.currBudget -= self.tradeCost
            self.currVal = (self.currQty * currPrice['close']).sum() + self.currBudget

            if (debug):
                # print currTime
                # print currPrice
                # print currOrder
                # print 'Trade Cost:', self.tradeCost
                # print 'Current Qty:', self.currQty
                print 'Current Budget:', self.currBudget
                print 'Current Val:', self.currVal
                # raw_input('WAIT')

            logList.append([currTime, currOrder, currPrice])

            bar.update(tradeId)

            tradeId += self.tradeFreq

            # Denotes whether we are currently holding any positions or not if daily trade
            # Helps in deciding how much to move at the end of the hold period 
            if (self.dailyTrade):
                if (currTimeStamp == highTime):
                    tradeId -= self.tradeFreq
                    tradeId += 1

            if (traceFlag):
                print 'loggingComputation: END'

        # Flush i.e. empty your positions
        if (self.flush):
            currTime = dataList['open'].iloc[endId - 1].name

            currPrice = pd.DataFrame(0, index = dataList['open'].columns, columns = ['open', 'close'])
            currPrice['open'] = dataList['open'].iloc[endId - 1]
            currPrice['close'] = dataList['close'].iloc[endId - 1]

            currOrder = pd.DataFrame(0, index = dataList['open'].columns, columns = ['signal', 'qty'])
            currOrder['signal'] = np.sign(-self.currQty)
            currOrder['qty'] = np.abs(-self.currQty)

            logList.append([currTime, currOrder, currPrice])

        bar.finish()

        print 'Finished running the strategy on', len(logList), 'timestamps'

        self.logOrders(logList)

        print 'FINAL STATS:'
        # print 'Trade Cost:', self.tradeCost
        # print 'Current Qty:', np.array(self.currQty)
        print 'Current Budget:', self.currBudget
        print 'Current Val:', self.currVal
        print 'NumTrades:', self.cntTrades
        print 'Avg. Profit:', (((self.currVal - self.initBudget) / 100.0) / (self.cntTrades + eps)) * 1e4


        if (debug):
            print 'Finished computing trades'
            raw_input('Finished Logging computed trades (Enter to continue):')


if __name__ == "__main__":

    # filesPath = '/home/nishantrai/Documents/data/Training_Data/Cash_data/'

    # stockList = getStockList(filesPath, year, numStocks)
    # stockList = ['ICICIBANK', 'HDFCBANK', 'YESBANK', 'FEDERALBNK', 'SBIN', 'INDUSINDBK', 'CANBK', 'KOTAKBANK', 'IDFCBANK', 'PNB']
    # stockList = ['CESC','DABUR','DRREDDY']
    # stockList = ['NIFTY', 'BANKNIFTY']
    # numStocks = len(stockList)

    config = {}
    config['INIT_BUDGET'] = 100000
    # config['PRICE_FILE_PATH']   = '/home/anishshah/Desktop/data/Dataset/Adjusted_FUT_Data/'
    config['PRICE_FILE_PATH'] = '/home/nishantrai/Documents/data/Training_Data/Adjusted_FUT_Data/'
    config['TRADE_FILE_PATH']   = None
    # config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA', 'NIFTY']
    # config['STOCK_LIST']      = ['AXISBANK', 'BANKBARODA', 'BANKINDIA', 'CANBK', 'HDFCBANK', 'ICICIBANK', 'INDUSINDBK', 'KOTAKBANK', 'PNB', 'SBIN', 'UNIONBANK']
    # NIFTY50: Works in 2010, 2012, 2014, 2016
    config['STOCK_LIST']      = ['ACC', 'AMBUJACEM', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BANKBARODA', 'BHARTIARTL', 'BHEL', 'BPCL',\
                                 'CIPLA', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDALCO',\
                                 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'ITC', 'KOTAKBANK', 'LT', 'LUPIN', 'M&M', 'MARUTI',\
                                 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBIN', 'SUNPHARMA', 'TATAMTRDVR', 'TATAPOWER', 'TATASTEEL',\
                                 'TCS', 'TECHM', 'ULTRACEMCO', 'WIPRO', 'ZEEL', 'CAIRN', 'PNB', 'NMDC', 'IDFC', 'DLF', 'JINDALSTEL']

    config['YEAR']       = 2010
    # config['STOCK_LIST'] = stockList
    config['START_TIME'] = dt2ut(datetime.datetime(config['YEAR'],1,1,0,0))
    config['END_TIME']   = dt2ut(datetime.datetime(config['YEAR'],12,31,0,0))
    config['FLUSH_AT_END'] = False
    config['DAILY_TRADE'] = True
    config["TRADE_FACT"]  = 0.1
    config['HOLD']        = 60
    config["HOLD_PERIOD"] = 5
    config["TRADE_FREQ"]  = 12
    config["HEDGE_METHOD"] = 'volBased'

    # Config for strategy
    config['WINDOW'] = 2
    config['INDEX']  = 'NIFTY'
    config['PERC_FLAG'] = True
    config['ABS_GAP_FLAG'] = True
    config['PERC_WINDOW']  = 25
    config['PERC_THRESHOLD']  = 0.5

    # Analyser flags
    config['MODE']            = 'percentile'
    config['ABS_FLAG']        = True
    config['BUCKET_SIZE']     = 20
    config['MIN_SIZE']        = 1500
    config['MAX_SIZE']        = 2000
    config['PLOT']            = False
    config['HEDGE']           = True
    config['HEDGE_STOCK']     = 'NIFTY'
    config['BETA_CORR_TYPE']  = 'constant'
    config['BETA_CORR']       = 1
    config["STOP_LOSS_VAL"]   = -0.005
    config["STOP_LOSS_FLAG"]  = False
    config["IGNORE_OUTLIERS"] = False
    config["HIGH_PERCENTILE"] = 0.99
    config["LOW_PERCENTILE"]  = 0.01

    config['DIVIDE_BY_VOLATILITY'] = False
    config['T_TEST_FLAG']     = True
    config['BINS']            = None

    # Useful utility flag for logging unique time stamps in each 
    # bucket, also for computing stddev, correlation, etc
    config['LOG_BUCKET_DATA'] = False

    #VWAP Window Parameters
    config['VWAP_GAP_WINDOW']     = 2
    config['VWAP_PRICE_WINDOW']   = 2

    # Only in the case of hedging
    config['STOCK_LIST'].append(config['INDEX'])

    sim = SimStrategy(config)
    sim.performAnalysis()
    sim.runStrategy()
