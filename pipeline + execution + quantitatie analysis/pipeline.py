from execStrategy import *
from backtester import *
from metrics import *
from visualise import *

debug = False

class Pipeline():

    def __init__(self, config):
        '''
        Initializes the Pipeline class object which is responsible for fitting
        everything together. It takes as input a config dictionary which is 
        further used to create the required objects
        '''

        self.config = config
        self.tradeListList = {}

    def combineDicts(self,dict1, dict2):
        for key in dict2:
            if key not in dict1:
                dict1[key] = dict2[key]
            else:
                dict1[key] = dict1[key].append(dict2[key])
        return dict1

    def getTrades(self, flushAtEnd = False, config = None):
        if (config is None):
            config = self.config

        config['FLUSH_AT_END'] = flushAtEnd

        sim = SimStrategy(config)            
        tradeList = sim.performAnalysis()
        # print(tradeList, type(tradeList))
        self.tradeListList = self.combineDicts(self.tradeListList, tradeList)
        sim.runStrategy()

    def printAllAnalyzeStats(self, flushAtEnd = False, config = None):

        sim = SimStrategy(config)
        sim.globalPrintStrategy(self.tradeListList)

    def backTestTrades(self, append = False, config = None):

        if (config is None):
            config = self.config

        config['APPEND'] = append

        print 'Starting backtesting'
        tester = backTester(self.config)
        print 'Initializing backtester finished'
        tester.backTest(tradeLog = False, verbose = False)
        print 'Backtesting finished'

        self.currBudget = tester.currVal

        if (debug):
            tester.printStats()

    def analyse(self, navId = ''):

        config = self.config

        print 'Starting analysis phase'
        statsDict = get_standard_ratios('results/backtestLog.csv', config)
        print_NAV_graph('results/backtestLog.csv', statsDict = statsDict, navId = navId)
        print 'Analysis phase finished'

def configCaller(fileName):

    import json

    with open(fileName) as jsonFile:    
        config = json.load(jsonFile)

    pip = Pipeline(config)

    periods = config['PERIOD_LIST']
    config['NUM_DAYS'] = 0

    for i in range(len(periods)):

        # Converting the strings to date time format
        period = map(pd.to_datetime, periods[i])

        config['YEAR']       = period[0].year
        config['START_TIME'] = dt2ut(period[0])
        config['END_TIME']   = dt2ut(period[1])
        config['NUM_DAYS']   += (datetime.fromtimestamp(config['END_TIME']) - datetime.fromtimestamp(config['START_TIME'])).days

        pip.getTrades(flushAtEnd = True, config = config)
        # The first one is not in append mode as it needs to write the headers
        # pip.backTestTrades(append = bool(i), config = config)

        # config['INIT_BUDGET'] = pip.currBudget

        if (debug):
            print 'Period:', period, 'done with budget', config['INIT_BUDGET']
    
    # pip.printAllAnalyzeStats(flushAtEnd = True, config = config)
    # pip.analyse()

if __name__ == '__main__':

    fileName = 'config/configGapUnified.json'

    configCaller(fileName)
