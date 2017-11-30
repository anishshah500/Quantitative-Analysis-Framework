import datetime
import pandas as pd
import csv
import numpy as np
from pandas.tseries.offsets import BDay
pd.options.mode.chained_assignment = None

debug = False

def dt2ut(dt):
    epoch = pd.to_datetime('1970-01-01')
    return (dt - epoch).total_seconds()

class getEarnings():

	def __init__(self,config):
		self.earningTimeStamps 		 = {}
		self.totalEarningsTimeStamps = {}

		self.earningsFile 	   = config['EARNINGS_FILE']
		self.stockList 		   = config['STOCK_LIST']
		self.version		   = config['VERSION'] 

		for stock in self.stockList:
			self.earningTimeStamps[stock] 		= []
			self.totalEarningsTimeStamps[stock] = []

	def isWithinBusinessHours(self, time):
		
		values = time.split(":")

		if((9 < int(values[0]) < 15) or (int(values[0]) == 9 and int(values[1]) >= 15) or (int(values[0]) == 15 and int(values[1]) <= 45)):
			return True

		else:
			return False

	def isOutsideBusinessHoursBeforeOpen(self, time):
		
		values = time.split(":")

		if((int(values[0]) < 9) or (int(values[0]) == 9 and int(values[1]) < 15)):
			return True

		else:
			return False

	def getEarningTimeStamps(self):
		
		earningsDF = pd.read_table(open(self.earningsFile),delimiter = "\t",names = ['date','time','stock'])
		
		earningsDFGroup = (earningsDF.groupby('stock'))

		for stock,group in earningsDFGroup:
			if stock in self.stockList:
				#DF op to convert into unix timestamp and set the time to 9 15 IST with timezone considered, +- 1 day
				if(self.version == 'TPlusMinus1'):
					group.loc[:,'date'] 		= group.loc[:,'date'].map(lambda x: (pd.to_datetime(str(x) + "034500", format='%Y%m%d%H%M%S')))
					group.loc[:,'date'] 		= group.loc[:,('date')].astype(np.int64) // 10 ** 9
					group.loc[:,'dateBefore']   = group.loc[:,('date')] - 86400
					group.loc[:,'nextDate']     = group.loc[:,('date')] + 86400

					# print(group['date'].values)
					self.earningTimeStamps[stock] = self.earningTimeStamps[stock] + list(group.loc[:,'date'])
					self.earningTimeStamps[stock] = self.earningTimeStamps[stock] + list(group.loc[:,'nextDate'])
					self.earningTimeStamps[stock] = self.earningTimeStamps[stock] + list(group.loc[:,'dateBefore'])
					self.earningTimeStamps[stock] = list(self.earningTimeStamps[stock])

				if(self.version == 'TPlus1'):
					group.loc[:,'date'] 		= group.loc[:,'date'].map(lambda x: (pd.to_datetime(str(x) + "034500", format='%Y%m%d%H%M%S')))
					group.loc[:,'date'] 		= group.loc[:,('date')].astype(np.int64) // 10 ** 9
					group.loc[:,'nextDate']     = group.loc[:,('date')] + 86400

					# print(group['date'].values)
					self.earningTimeStamps[stock] = self.earningTimeStamps[stock] + list(group.loc[:,'date'])
					self.earningTimeStamps[stock] = self.earningTimeStamps[stock] + list(group.loc[:,'nextDate'])
					self.earningTimeStamps[stock] = list(self.earningTimeStamps[stock])

				if(self.version == 'withinBusinessDays' or self.version == 'outsideBusinessDays'):
					
					currGroupWithin 				   = group[group['time'].map(self.isWithinBusinessHours) == True]
					
					if(debug):
						print(stock)
						print(currGroupWithin)
					
					currGroupWithin.loc[:,'date'] 	   = currGroupWithin.loc[:,'date'].map(lambda x: \
						(pd.to_datetime(str(x) + "034500", format='%Y%m%d%H%M%S') + BDay(1)))
					
					currGroupWithin.loc[:,'date']      = currGroupWithin.loc[:,'date'].map(dt2ut).astype(np.int64)
					
					if(self.version == 'withinBusinessDays'):
						self.earningTimeStamps[stock] 	   = list(currGroupWithin.loc[:,'date'])
					
					if(debug):
						print(self.earningTimeStamps[stock])

					self.totalEarningsTimeStamps[stock] += list(currGroupWithin.loc[:,'date'])

					currGroupOutside 					= group[group['time'].map(self.isWithinBusinessHours) == False]
					currGroupOutsideBeforeOpen			= currGroupOutside[currGroupOutside['time'].map(self.isOutsideBusinessHoursBeforeOpen) == True]
					currGroupOutsideAfterClose			= currGroupOutside[currGroupOutside['time'].map(self.isOutsideBusinessHoursBeforeOpen) == False]
					
					if(debug):
						print(stock)
						print(currGroupOutside)
					
					currGroupOutsideBeforeOpen.loc[:,'date'] 		= currGroupOutsideBeforeOpen.loc[:,'date'].map(lambda x: \
						(pd.to_datetime(str(x) + "034500", format='%Y%m%d%H%M%S')))

					currGroupOutsideAfterClose.loc[:,'date'] 		= currGroupOutsideAfterClose.loc[:,'date'].map(lambda x: \
						(pd.to_datetime(str(x) + "034500", format='%Y%m%d%H%M%S') + BDay(1)))
					
					currGroupOutsideBeforeOpen.loc[:,'date']      = currGroupOutsideBeforeOpen.loc[:,'date'].map(dt2ut).astype(np.int64)
					currGroupOutsideAfterClose.loc[:,'date']      = currGroupOutsideAfterClose.loc[:,'date'].map(dt2ut).astype(np.int64)
					
					if(self.version == 'outsideBusinessDays'):
						self.earningTimeStamps[stock] 		= list(currGroupOutsideBeforeOpen.loc[:,'date']) + list(currGroupOutsideAfterClose.loc[:,'date'])
					
					if(debug):
						print(self.earningTimeStamps[stock])

					self.totalEarningsTimeStamps[stock] += list(currGroupOutsideBeforeOpen.loc[:,'date']) 
					self.totalEarningsTimeStamps[stock] += list(currGroupOutsideAfterClose.loc[:,'date'])

if __name__ == '__main__':

	config = {}

	config['EARNINGS_FILE'] = '/home/anishshah/Desktop/data/Earnings_dates'
	config['VERSION']		= "outsideBusinessDays"
	config['STOCK_LIST']	= ["ACC", "AMBUJACEM", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BANKBARODA", "BHARTIARTL", "BHEL", "BPCL",\
                  "CIPLA", "DRREDDY", "GAIL", "GRASIM", "HCLTECH", "HDFC", "HDFCBANK", "HINDALCO",\
                  "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "ITC", "KOTAKBANK", "LT", "LUPIN", "M&M", "MARUTI",\
                  "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBIN", "SUNPHARMA", "TATAMTRDVR", "TATAPOWER", "TATASTEEL",\
                  "TCS", "TECHM", "ULTRACEMCO", "WIPRO", "ZEEL", "CAIRN", "PNB", "NMDC", "IDFC", "DLF", "JINDALSTEL", "NIFTY"]
	
	earnings = getEarnings(config)
	earnings.getEarningTimeStamps()
	print(earnings.earningTimeStamps)
	print(earnings.totalEarningsTimeStamps)