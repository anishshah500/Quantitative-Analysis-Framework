######################################################################
def avgwin(x):
	return np.mean([y for y in x if y>0])
######################################################################
def avgloss(x):
    return np.mean([y for y in x if y<0])
######################################################################
def winratio(x):
    return avgwin(x) / avgloss(x)
######################################################################
def accuracy(x):
    return 100*len([y for y in x if y>0]) / len(x)
######################################################################
grandDF['profitbps_60'] = grandDF['profit_60'] * 10000
######################################################################
grandDF.pivot_table(index ='finalCategory', values = 'profitbps_60', aggfunc = [np.mean, len, accuracy, winratio, avgwin, avgloss,max,min])
######################################################################
def filterdf(df, f, col):
    return df[f(df, col)]
######################################################################
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegressionCV(random_state = 0, tol = 1e-5)
clf = DecisionTreeClassifier(random_state=0, class_weight = {-1:6, 0:1, 1:6}, max_depth = 2,\
 min_samples_split = 100, min_samples_leaf = 50, min_impurity_split = 0.3)
clf.fit(X_train, y_train)
clf.coef_
print clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print classification_report(y_test, y_pred)	
#####################################################################
for hold in holds:        
    print((testDf.groupby(['gapFilled_'+str(hold),'binID']).count() / testDf.groupby('binID').count()).iloc[:,1])
    (testDf.groupby(['gapFilled_'+str(hold),'binID']).count() / testDf.groupby('binID').count()).iloc[:,1].to_clipboard
    input()
#####################################################################
for hold in holds:
	graphData.append((len(testDf[testDf['gapFilled_' + str(hold)] == True]) / len(testDf)))
#####################################################################
import matplotlib.pyplot as plt
plt.plot(graphData)
plt.ylabel('fractions of gap filled')
plt.xlabel('hold in minutes')
plt.show()
#####################################################################
for ID in range(min(testDf['binID']),max(testDf['binID']) + 1):
    graphData = []
    for hold in holds:
        graphData.append((len(testDf[(testDf['gapFilled_' + str(hold)] == True) & (testDf['binID'] == ID)])) / len(testDf))
    plt.plot(graphData)
    plt.ylabel('fractions of gap filled')
    plt.xlabel('hold in minutes, binID = ' + str(ID))
    plt.show()
    input()
#####################################################################
for ID in range(min(testDf['binID']),max(testDf['binID']) + 1):
     graphData = []
     for hold in holds:
         graphData.append((len(testDf[(testDf['gapFilled_' + str(hold)] == True) & (testDf['binID'] == ID)])) / len(testDf[testDf['binID'] == ID]))
     mainGraphData.append(graphData)
#####################################################################
for i in range(len(mainGraphData)):
     plt.plot(holds,mainGraphData[i],label = i)

plt.legend(bbox_to_anchor=(1, 1))
plt.show()
#####################################################################
((testDf.loc[testDf['gapFilled' + '_' + str(60)] == True,'currOpen'] / testDf.loc[testDf['gapFilled' + '_' + str(60)] == True,'prevClose'] - 1)\
 * -np.sign(testDf.loc[testDf['gapFilled' + '_' + str(60)] == True,'signal'])).mean() * 10000
#####################################################################
#target implementation
holds = list(range(2,360))
for hold in holds:
    testDf.loc[testDf['gapFilled' + '_' + str(hold)] == True,'profit' + '_' + str(hold)] = \
    ((testDf.loc[testDf['gapFilled' + '_' + str(hold)] == True,'entryPrice']\
     / testDf.loc[testDf['gapFilled' + '_' + str(hold)] == True,'prevClose'] - 1) * \
    -np.sign(testDf.loc[testDf['gapFilled' + '_' + str(hold)] == True,'signal']))
#####################################################################
for key, gp in testDf.groupby('binID'):
    print(len(gp[gp['profit_360'] >= 0]) / len(gp))
#####################################################################
def loadAnalyserResults(yearList):
    '''
    Takes a list of years as input and loads the
    result files into it.
    '''

    yearStats = {}
    for year in yearList:
#        print 'Currently at year:', year
        with open('results/gapStatsDf' + str(year) + '.pickle', 'rb') as handle:
            yearStats[year] = pickle.load(handle, encoding = 'latin1')

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
        grandStats = grandStats.append(deepcopy(tmpDf))

#    print 'Loading results complete!'

    return grandStats
#####################################################################
for param in params:
    print(param)
    print("neg")
    print(len(testDf[(testDf['gapRatio'] < param) & (testDf['gapFilled_60'] == True)]) / len(testDf[testDf['gapRatio'] < param]))
    print("pos")
    print(len(testDf[(testDf['gapRatio'] >= param) & (testDf['gapFilled_60'] == True)]) / len(testDf[testDf['gapRatio'] >= param]))
    print("\n")
#####################################################################
max = np.amax(test['profit_60'])
min = np.amin(test['profit_60'])
histo = np.histogram(test['profit_60'], bins = 100, range = (min,max))
freqs = histo[0]
rangebins = (max - min)
numberbins = (len(histo[1])-1)
interval = (rangebins/numberbins)
newbins = np.arange((min), (max), interval)
histogram = plt.bar(newbins, freqs, width=0.001)
plt.show()
#####################################################################
def getFetRank(fet, grandStats):
    return grandStats.groupby(grandStats.index)[fet].rank(pct = True) * 100.0
#####################################################################
test = testDf[testDf.index.map(lambda x: datetime.datetime.fromtimestamp(x).year) == 2010]
test.groupby('stockName').mean()['profit_60'] 
testDf.groupby('stockName').mean()['profit_60']
#####################################################################
testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4)]['profit_60'].mean()
#####################################################################
pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4)], \
    index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#####################################################################
#feature1 - gapRatio
pivot_table(testDf[testDf['gapRatio'] <= -1], index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#feature2 - volDailyRank
pivot_table(testDf[testDf['volDailyRank'] > 70], index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#feature3 - openInHighLow
pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4)], \
    index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#feature 1+2
pivot_table(testDf[(testDf['volDailyRank'] > 70) & (testDf['gapRatio'] <= -1)], index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#feature 1+3
pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4) & \
    (testDf['gapRatio'] <= -1)], index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#feature 2+3
pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4) & \
    (testDf['volDailyRank'] > 70)], index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#feature 1+2+3
pivot_table(self.dataDF[((self.dataDF['isInsideGap'] == True) & (self.dataDF['openInLowHigh'] >= 0.2) & (self.dataDF['openInLowHigh'] < 0.4)) \
        & (self.dataDF['volDailyRank'] > 70) & (self.dataDF['gapRatio'] <= -1)], index = 'binID', values = 'profit_60', aggfunc = [len,np.mean])
#####################################################################
def getData(holds):

    self.dataDF = self.dataDF.sort_index()
    self.dataDF['volDailyRank'] = getFetRank('vol', self.dataDF)

    self.dataDF['openInLowHighInt'] = self.dataDF['openInLowHigh'] * 1.0
        
    for hold in holds:
        #feature1 - gapRatio
        print(pivot_table(testDf[testDf['gapRatio'] <= -1], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]))
        pivot_table(testDf[testDf['gapRatio'] <= -1], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]).to_clipboard()
        input("Continue...")
        #feature2 - volDailyRank
        print(pivot_table(testDf[testDf['volDailyRank'] > 70], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]))
        pivot_table(testDf[testDf['volDailyRank'] > 70], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]).to_clipboard()
        input("Continue...")
        #feature3 - openInHighLow
        pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4)], \
            index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]).to_clipboard()
        print(pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4)], \
            index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]))
        input("Continue...")
        #feature 1+2
        pivot_table(testDf[(testDf['volDailyRank'] > 70) & (testDf['gapRatio'] <= -1)], index = 'binID',\
         values = 'profit_' + str(hold), aggfunc = [len,np.mean]).to_clipboard()
        print(pivot_table(testDf[(testDf['volDailyRank'] > 70) & (testDf['gapRatio'] <= -1)], index = 'binID',\
         values = 'profit_' + str(hold), aggfunc = [len,np.mean]))
        input("Continue...")
        #feature 1+3
        print(pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4) & \
            (testDf['gapRatio'] <= -1)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]))
        pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4) & \
            (testDf['gapRatio'] <= -1)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]).to_clipboard()
        input("Continue...")
        #feature 2+3
        print(pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4) & \
            (testDf['volDailyRank'] > 70)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]))
        pivot_table(testDf[(testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4) & \
            (testDf['volDailyRank'] > 70)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]).to_clipboard()
        input("Continue...")
        #feature 1+2+3
        print(pivot_table(testDf[((testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4)) \
                & (testDf['volDailyRank'] > 70) & (testDf['gapRatio'] <= -1)], index = 'binID', values = 'profit_' + str(hold), aggfunc = [len,np.mean]))
        pivot_table(testDf[((testDf['isInsideGap'] == True) & (testDf['openInLowHigh'] >= 0.2) & (testDf['openInLowHigh'] < 0.4)) \
                & (testDf['volDailyRank'] > 70) & (testDf['gapRatio'] <= -1)], index = 'binID', values = 'profit_' + str(hold), \
                aggfunc = [len,np.mean]).to_clipboard()
