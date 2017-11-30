import numpy as np
import pandas as pd
import pickle
from ttest import *
from analyser import *
from earningDates import *
from pandas import pivot_table

debug = False

eps = 1e-10

def avgwin(x):
    return np.mean([y for y in x if y>0])

def avgloss(x):
    return np.mean([y for y in x if y<0])

def winratio(x):
    return avgwin(x) / avgloss(x)

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
    results['stDev'] = np.std(profit)

    return results

class compileResults():

	def __init__(self):
		self.dataDF = pd.DataFrame()
		self.holdPeriodList = [30,60]

	def loadAnalyserResults(self,yearList, suffix = 'noStopLoss_full_360_withOutlier_newVol', pricePerc = True):
	    '''
	    Takes a list of years as raw_input and loads the
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

	    self.dataDF = grandStats

	def calculateAndPrintStats(self,fet = 'profit'):
		print (' '.join(['*']*50))
		print 'Global Statistics: Cumulative over all years'
		print (' '.join(['*']*50))
		for hold in self.holdPeriodList:
			currResult = pivot_table(self.dataDF, index = 'binID', values = fet + '_' + str(hold), aggfunc = [len, accuracy, avgwin, avgloss, np.min, \
				np.max, winratio, np.mean, np.std])

			print(currResult)
			currResult.to_clipboard()
			raw_input('Continue..')
			print (' '.join(['*']*50))

		print (' '.join(['*']*50))
		print('Bucket Stats')
		print (' '.join(['*']*50))
		currResult = pivot_table(self.dataDF, index = 'binID', values = 'gapRatio', aggfunc = [len,np.mean,np.median,np.std,np.max,np.min])
		print(currResult)
		currResult.to_clipboard()
		raw_input('Continue..')

	def performTTest(self, posSample, negSample, hold, printResult = True):
	    '''
	    Performs the t test and returns the results,
	    feature represents the attribute to be considered,
	    It is actually a dataframe of the concerned values
	    cond is a function which divides the set into two
	    splits
	    '''

	    results = {}

	    # posSample = tradeDf[feature.map(cond)]
	    # negSample = tradeDf[~ feature.map(cond)]

	    results['pos'] = pd.DataFrame(getStats(posSample['profit' + '_' + str(hold)]), index = ['pos'])
	    results['neg'] = pd.DataFrame(getStats(negSample['profit' + '_' + str(hold)]), index = ['neg'])
	    results['tot'] = pd.DataFrame(getStats(self.dataDF['profit' + '_' + str(hold)]), index = ['tot'])

	    # results['pos']['gapFilled'] = len(posSample[posSample['gapFilled' + '_' + str(hold)] == True]) / float(len(posSample))
	    # results['neg']['gapFilled'] = len(negSample[negSample['gapFilled' + '_' + str(hold)] == True]) / float(len(negSample))
	    # results['tot']['gapFilled'] = len(tradeDf[tradeDf['gapFilled' + '_' + str(hold)] == True]) / float(len(tradeDf))

	    # results = pd.DataFrame(results).transpose()
	    ttestResult = stats.ttest_ind(posSample['profit' + '_' + str(hold)], negSample['profit' + '_' + str(hold)])

	    if(printResult):
	    	print(' '.join(['*'] * 50))
	    	print("Results")
	    	print(pd.concat([results['neg'],results['pos'],results['tot']]))
	    	pd.concat([results['neg'],results['pos'],results['tot']]).to_clipboard()
	    	raw_input('Continue...')
	    	print(' '.join(['*'] * 50))
	    	print("T Test Stats")
	    	print(pd.DataFrame([ttestResult[0],ttestResult[1]], index = ['tTestVal', 'pVal']).T)
	    	pd.DataFrame([ttestResult[0],ttestResult[1]], index = ['tTestVal', 'pVal']).T.to_clipboard()
	    	raw_input('Continue...')

	def tTestEarnings(self, config):
			
			earnings = getEarnings(config)
			earnings.getEarningTimeStamps()
			
			dataDFWithoutEarnings = pd.DataFrame()
			dataDFEarnings 		  = pd.DataFrame()

			for stock in earnings.earningTimeStamps:
				
				currStockDF 			   = self.dataDF[self.dataDF['stockName'] == stock]
				currStockDFWithoutEarnings = currStockDF[~currStockDF.index.isin(earnings.totalEarningsTimeStamps[stock])]
				currStockDFEarnings 	   = currStockDF[currStockDF.index.isin(earnings.earningTimeStamps[stock])]
				dataDFWithoutEarnings 	   = dataDFWithoutEarnings.append(currStockDFWithoutEarnings)
				dataDFEarnings             = dataDFEarnings.append(currStockDFEarnings)
				
				if(debug):
					print(' '.join(['*']*50))
					print(stock)
					print(dataDFWithoutEarnings)
					print(dataDFEarnings)

			self.performTTest(dataDFWithoutEarnings,dataDFEarnings,60)
			print(np.mean(np.abs(dataDFWithoutEarnings['gapRatio'])), np.mean(np.abs(dataDFEarnings['gapRatio'])),np.mean(np.abs(self.dataDF['gapRatio'])))
			print(np.std(dataDFWithoutEarnings['gapRatio']), np.std(dataDFEarnings['gapRatio']), np.std(self.dataDF['gapRatio']))

	def customTTest(self, gapRatio = True, volDailyRank = True, openInLowHighInt = True):

		self.dataDF = self.dataDF.sort_index()
		self.dataDF['volDailyRank'] = getFetRank('vol', self.dataDF)

		self.dataDF['openInLowHighInt'] = self.dataDF['openInLowHigh'] * 1.0

		#combinations

		#1filter combinations
		posSample = self.dataDF[(self.dataDF['gapRatio'] <= -1)]
		negSample = self.dataDF[~((self.dataDF['gapRatio'] <= -1))]

		# posSample = self.dataDF[(self.dataDF['volDailyRank'] > 70)]
		# negSample = self.dataDF[~((self.dataDF['volDailyRank'] > 70))]

		#2 filter combinations
		# posSample = self.dataDF[(self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)]
		# negSample = self.dataDF[~((self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1))]

		# posSample = self.dataDF[((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & (self.dataDF['openInLowHigh'] < 0.4)) \
		# & (self.dataDF['gapRatio'] <= -1)]
		# negSample = self.dataDF[~(((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & (self.dataDF['openInLowHigh'] < 0.4)) & \
		# 	(self.dataDF['gapRatio'] <= -1))]

		# posSample = self.dataDF[((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & (self.dataDF['openInLowHigh'] < 0.4)) \
		# & (self.dataDF['volDailyRank'] > 70)]
		# negSample = self.dataDF[~(((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & (self.dataDF['openInLowHigh'] < 0.4)) & \
		# 	(self.dataDF['volDailyRank'] > 70))]

		#All 3 filters

		# posSample = self.dataDF[((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & (self.dataDF['openInLowHigh'] < 0.4)) \
		# & (self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)]
		# negSample = self.dataDF[~(((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & (self.dataDF['openInLowHigh'] < 0.4)) & \
		# 	(self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1))]
		
		# posSample = self.dataDF[(self.dataDF['openInLowHighInt'] <= -0.01) & (self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)]
		# negSample = self.dataDF[~((self.dataDF['openInLowHighInt'] <= -0.01) & (self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1))]

		for hold in [60,120,240,360]:
			print(hold)
			self.performTTest(posSample,negSample,hold)

	def getFilterStats(self):

		self.dataDF.sort_index()
		self.dataDF['volDailyRank'] = getFetRank('vol', self.dataDF)

		noFiltersExp      = self.dataDF.groupby('binID')['profit_60'].mean()
		gapRatioFilterExp = self.dataDF[self.dataDF['gapRatio'] <= -1].groupby('binID')['profit_60'].mean()
		volatilityRankExp = self.dataDF[self.dataDF['volDailyRank'] > 70].groupby('binID')['profit_60'].mean()
		gapVolFilterExp   = self.dataDF[(self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)].groupby('binID')['profit_60'].mean()

	def getData(self,holds):

	    for hold in holds:
	        
	        lenDF 	 = pd.DataFrame()
	        profitDF = pd.DataFrame()

	        self.dataDF = self.dataDF.sort_index()
	        self.dataDF['volDailyRank'] = getFetRank('vol', self.dataDF)
	        self.dataDF['openInLowHighInt'] = self.dataDF['openInLowHigh'] * 1.0

	        #no feature
	        lenDF = lenDF.append(pivot_table(self.dataDF, index = 'binID', values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF, index = 'binID', values = 'profit_' + str(hold), aggfunc = [np.mean]).T)

	        #feature1 - gapRatio
	        lenDF = lenDF.append(pivot_table(self.dataDF[self.dataDF['gapRatio'] <= -1], index = 'binID', \
	        	values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF[self.dataDF['gapRatio'] <= -1], index = 'binID', \
	        	values = 'profit_' + str(hold), aggfunc = [np.mean]).T)
	        
	        #feature2 - volDailyRank
	        lenDF = lenDF.append(pivot_table(self.dataDF[self.dataDF['volDailyRank'] > 70], index = 'binID', \
	        	values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF[self.dataDF['volDailyRank'] > 70], index = 'binID', \
	        	values = 'profit_' + str(hold), aggfunc = [np.mean]).T)
	        
	        #feature3 - openInHighLow
	        lenDF = lenDF.append(pivot_table(self.dataDF[(self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) \
	        	& (self.dataDF['openInLowHigh'] < 0.4)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF[(self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) \
	        	& (self.dataDF['openInLowHigh'] < 0.4)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [np.mean]).T)
	        
	        #feature 1+2
	        lenDF = lenDF.append(pivot_table(self.dataDF[(self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)], index = 'binID',\
	         values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF[(self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)], index = 'binID',\
	         values = 'profit_' + str(hold), aggfunc = [np.mean]).T)
	        
	        #feature 1+3
	        lenDF = lenDF.append(pivot_table(self.dataDF[(self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & \
	        	(self.dataDF['openInLowHigh'] < 0.4) &  (self.dataDF['gapRatio'] <= -1)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF[(self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & \
	        	(self.dataDF['openInLowHigh'] < 0.4) &  (self.dataDF['gapRatio'] <= -1)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [np.mean]).T)
	        
	        #feature 2+3
	        lenDF = lenDF.append(pivot_table(self.dataDF[(self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & \
	        	(self.dataDF['openInLowHigh'] < 0.4) &
	         (self.dataDF['volDailyRank'] > 70)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF[(self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & \
	        	(self.dataDF['openInLowHigh'] < 0.4) &
	         (self.dataDF['volDailyRank'] > 70)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [np.mean]).T)
	        
	        #feature 1+2+3
	        lenDF = lenDF.append(pivot_table(self.dataDF[((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) \
	        	& (self.dataDF['openInLowHigh'] < 0.4)) & (self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)], \
	                index = 'binID', values = 'profit_' + str(hold), aggfunc = [len]).T)
	        profitDF = profitDF.append(pivot_table(self.dataDF[((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) \
	        	& (self.dataDF['openInLowHigh'] < 0.4)) & (self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)], \
	                index = 'binID', values = 'profit_' + str(hold), aggfunc = [np.mean]).T)

	        print(''.join(['*']*50))	        
	        print("Filter 1 : gapRatio <= -1")
	        print("Filter 2 : daily relative volatility rank > 70")
	        print("Filter 3 : open in low high metric in [0.2,0.4)")
	        print("Hold = " + str(hold))

	        lenDF.set_index(pd.Series(['noFilters','1','2','3','1+2','1+3','2+3','1+2+3']), inplace = True)
	        profitDF.set_index(pd.Series(['noFilters','1','2','3','1+2','1+3','2+3','1+2+3']), inplace = True)

	        print(''.join(['*']*50))
	        print("Number of records")
	        print(lenDF)
	        lenDF.to_clipboard()
	        raw_input('Continue...')

	        print(''.join(['*']*50))
	        print("Expectancy")
	        print(profitDF)
	        profitDF.to_clipboard()
	        raw_input('Continue...')

if __name__ == "__main__":
	
	config = {}

	config['EARNINGS_FILE'] = '/home/anishshah/Desktop/data/Earnings_dates'
	config['VERSION']		= "withinBusinessDays"
	config['STOCK_LIST']	= ["ACC", "AMBUJACEM", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BANKBARODA", "BHARTIARTL", "BHEL", "BPCL",\
                  "CIPLA", "DRREDDY", "GAIL", "GRASIM", "HCLTECH", "HDFC", "HDFCBANK", "HINDALCO",\
                  "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "ITC", "KOTAKBANK", "LT", "LUPIN", "M&M", "MARUTI",\
                  "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBIN", "SUNPHARMA", "TATAMTRDVR", "TATAPOWER", "TATASTEEL",\
                  "TCS", "TECHM", "ULTRACEMCO", "WIPRO", "ZEEL", "CAIRN", "PNB", "NMDC", "IDFC", "DLF", "JINDALSTEL", "NIFTY"]

	compileObject = compileResults()
	compileObject.loadAnalyserResults([2010,2012,2014,2016],suffix = 'noStopLoss_full_360_noOutlier_newVol')
	# print compileObject.dataDF
	compileObject.calculateAndPrintStats(fet = 'profitPerVol')
	# compileObject.tTestEarnings(config)
	# compileObject.customTTest()
	# compileObject.getFilterStats()
	# compileObject.getData([60,120,240,360])