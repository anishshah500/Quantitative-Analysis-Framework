from ttest import *

pd.options.mode.chained_assignment = None

def getOptimalBoundary(profitDf, minFrac = 0.1):
    '''
    Find optimal boundary on the basis of the average profit,
    while ensuring number of samples is >= minFrac
    '''

    tmpDf = profitDf[profitDf['leftNum'] >= minFrac]
    boundary = tmpDf['leftAvgProfit'].idxmax()
    return boundary

def getFetStats(fet, tradeDf, minFrac = 0.1, minNum = 100, jump = 10, plot = True):

    profitDf = tradeDf.set_index(fet)
    profitDf = profitDf.sort_index()
    profitDf['leftNum'] = range(1, profitDf.shape[0] + 1)
    profitDf['rightNum'] = range(profitDf.shape[0], 0, -1)
    profitDf['profit'] *= 10000
    profitDf['forwardCumSum'] = profitDf['profit'].cumsum()
    profitDf['reverseCumSum'] = profitDf['profit'].ix[::-1].cumsum()[::-1]
    profitDf['leftAvgProfit'] = profitDf['forwardCumSum'] / profitDf['leftNum']
    profitDf['rightAvgProfit'] = profitDf['reverseCumSum'] / profitDf['rightNum']

    # print profitDf.loc[:, ['profit', 'leftNum', 'leftAvgProfit', 'rightAvgProfit']]

    totLen = profitDf.shape[0]
    profitDf['leftNum'] /= (totLen * 1.0)
    profitDf['rightNum'] /= (totLen * 1.0)

    boundary = getOptimalBoundary(profitDf, minFrac = minFrac)

    plotIdx = range(minNum, (totLen - minNum), jump)
    # profitDf.loc[:, ['leftAvgProfit', 'rightAvgProfit']].iloc[range(minNum, (totLen - minNum), jump)].plot(colormap = 'gist_rainbow')
    # plt.show()

    corr = np.corrcoef(profitDf['profit'], profitDf.index)[0][1]

    if (plot):
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        profitDf.loc[:, ['leftAvgProfit', 'rightAvgProfit']].iloc[plotIdx].plot(ax = ax, colormap = 'gist_rainbow')
        profitDf.loc[:, ['leftNum', 'rightNum']].iloc[plotIdx].plot(ax = ax2, secondary_y = True)
        plt.show()

    return {'boundary': boundary, 'corr': corr}

def getFetStatsList(fet, holdGrandStats, holdList, minFrac = 0.1, minNum = 100, jump = 10, plot = True):

    fig, axes = plt.subplots(nrows = 2, ncols = 2, sharey = True)
    fig.subplots_adjust(hspace=0.25)

    for i in range(len(holdList)):

        hold = holdList[i]

        tradeDf = holdGrandStats[holdGrandStats['hold'] == hold]

        profitDf = tradeDf.set_index(fet)
        profitDf = profitDf.sort_index()
        profitDf['leftNum'] = range(1, profitDf.shape[0] + 1)
        profitDf['rightNum'] = range(profitDf.shape[0], 0, -1)
        profitDf['profit'] *= 10000
        profitDf['forwardCumSum'] = profitDf['profit'].cumsum()
        profitDf['reverseCumSum'] = profitDf['profit'].ix[::-1].cumsum()[::-1]
        profitDf['leftAvgProfit'] = profitDf['forwardCumSum'] / profitDf['leftNum']
        profitDf['rightAvgProfit'] = profitDf['reverseCumSum'] / profitDf['rightNum']

        # print profitDf.loc[:, ['profit', 'leftNum', 'leftAvgProfit', 'rightAvgProfit']]

        totLen = profitDf.shape[0]
        profitDf['leftNum'] /= (totLen * 1.0)
        profitDf['rightNum'] /= (totLen * 1.0)

        boundary = getOptimalBoundary(profitDf, minFrac = minFrac)

        plotIdx = range(minNum, (totLen - minNum), jump)
        # profitDf.loc[:, ['leftAvgProfit', 'rightAvgProfit']].iloc[range(minNum, (totLen - minNum), jump)].plot(colormap = 'gist_rainbow')
        # plt.show()

        corr = np.corrcoef(profitDf['profit'], profitDf.index)[0][1]

        if (plot):
            ax=axes[i/2, i%2]
            ax2 = ax.twinx()
            profitDf.loc[:, ['leftAvgProfit', 'rightAvgProfit']].iloc[plotIdx].plot(ax = ax, colormap = 'gist_rainbow',\
                             title = 'Variation of payoff with ' + fet + ' for hold: ' + str(hold), alpha = 0.5, grid = True)
            # profitDf.loc[:, ['leftNum', 'rightNum']].iloc[plotIdx].plot(ax = ax2, secondary_y = False, alpha = 0.5)
            ax.legend(('avgProfit: ' + fet + ' < x', 'avgProfit: ' + fet + ' >= x'), loc = 'lower left')
            ax2.legend(('Fraction: ' + fet + ' < x', 'Fraction: ' + fet + ' >= x'), loc = 'lower right')

    plt.suptitle('Payoff (In bps) Variation with -' + fet)
    plt.show()

    return {'boundary': boundary, 'corr': corr}

def getTimeStats(tradeDf, origTrades):

    # Preapre the groupings
    dayWiseTrade = tradeDf.groupby(tradeDf.index)
    longTrades   = tradeDf[tradeDf['gapRatio'] <= 0]
    longTrades   = longTrades.groupby(longTrades.index)
    shortTrades  = tradeDf[tradeDf['gapRatio'] > 0]
    shortTrades  = shortTrades.groupby(shortTrades.index)

    # Get the count statistics
    numTrades = pd.DataFrame(dayWiseTrade['gapRatio'].count())
    numTrades.columns = ['count']
    numTrades['shortCount']  = (shortTrades['gapRatio'].count())
    numTrades['longCount']   = (longTrades['gapRatio'].count())
    numTrades.fillna(0, inplace = True)
    numTrades['netTrades'] = numTrades['longCount'] - numTrades['shortCount']

    numTrades = numTrades.reindex(set(origTrades.index)).fillna(0).sort_index()
    numTrades.index = pd.to_datetime(numTrades.index, unit = 's')

    print numTrades.describe()
    numTrades.describe().to_clipboard()
    # numTrades.plot(colormap = 'gist_rainbow', alpha = 0.5)
    # plt.show()

if __name__ == '__main__':

    # yearList = [2016]
    yearList = [2010,2012,2014,2016]
    # holdList = (range(3, 30, 3) + range(30, 120, 6) + range(120, 301, 12))
    holdList = [60,120,240,360]
    # holdList = [60]
    # Compute daily gap Rank
    weekDay = False
    gapZone = False
    gapRank = True
    volRank = True
    notional = False
    rangeFlag = True
    volumeRank= True

    colList = [u'currOpen', u'prevClose', u'entryPrice', u'gapSize', u'dailyRet',
                u'signal', u'vol', u'gapRatio', u'percentile', u'round_pcile', u'binID',                                                                                                                        
                u'stockName', u'beta', u'gapRatioIndex', u'prevLow', u'prevHigh', u'prevOpen',
                u'notionalPerc', u'pricePerc']


    grandStats = loadAnalyserResults(yearList, suffix = 'noStopLoss_full_360_noOutlier_newVol')

    if (weekDay):
        grandStats['weekDay'] = getWeekDay(grandStats)
        colList.append('weekDay')

    if (gapZone):
        grandStats = grandStats[grandStats['prevDayRet'] > 0]
        grandStats['gapZone'] = getGapZone(grandStats)
        colList.append('gapZone')

    if (gapRank):
        grandStats = grandStats.sort_index()
        grandStats['gapDailyRank'] = getFetRank('gapRatio', grandStats)
        colList.append('gapDailyRank')

    if (volRank):
        grandStats = grandStats.sort_index()
        grandStats['volDailyRank'] = getFetRank('vol', grandStats)
        # Invert to maintain the negative correlation
        # grandStats['volDailyRank'] = 100.0 - grandStats['volDailyRank']
        colList.append('volDailyRank')

    if (rangeFlag):
        grandStats['openInLowHighInt'] = grandStats['openInLowHigh'] * 1.0
        colList.append('openInLowHighInt')
        grandStats = grandStats[(0.0 <= grandStats['openInLowHighInt'])  & (grandStats['openInLowHighInt'] < 1.0)]

    if (notional):
        grandStats['notionalPerc'] = 100.0 - grandStats['notionalPerc']

    if(volumeRank):
        grandStats = grandStats.sort_index()
        grandStats['volumeDailyRank'] = getFetRank('volumeCurrOpen', grandStats)
        colList.append('volumeDailyRank')

    # fetStats = (grandStats['notionalPerc']).describe()
    # print fetStats

    grandStats['vol'] = 1.0 - grandStats['vol']

    holdGrandStats = getHoldWiseStats(grandStats, holdList, colList)

    fact = 0.1
    minFrac = [fact, fact, fact, fact]
    # fetList = ['gapRatio', 'volDailyRank', 'openInLowHighInt', 'notionalPerc', 'pricePerc']
    # fetList = ['gapRatio', 'volDailyRank', 'openInLowHighInt']
    fetList = ['openInLowHighInt']
    bList   = [-0.65, 13, -0.01]
    # fetList = ['notionalPerc']
    weightList = [1.0, 1.0, 1.0]

    print 'Minimum fraction to be kept in each individual model:', minFrac

    statList = {}
    fetStatList = {}

    # getFetStatsList('vol', holdGrandStats, holdList, minFrac = fact, plot = True)
    # raw_input('waiting..')

    for hold in holdList:

        fetStatList[hold] = {}

        tradeDf = holdGrandStats[holdGrandStats['hold'] == hold]
        tradeDf['score'] = 0.0

        for i in range(len(fetList)):

            fet = fetList[i]

            getFetStatsList(fet, holdGrandStats, holdList, minFrac = fact, plot = True)

            # results = getFetStats(fet, tradeDf, minFrac = minFrac[i], plot = True)
            # boundary = results['boundary']
            # boundary = bList[i]
            # fetStatList[hold][fet] = results
            # dev = np.std(tradeDf[fet] - boundary)
            # fetStatList[hold][fet]['dev'] = dev
            # Features to consider, gapRatio, openInLowHighInt, volDailyRank, (gapSize)
            # print ((tradeDf[fet] - boundary) / dev)
            # tmpScore = ((boundary - tradeDf[fet]) / dev)
            # tmpScore[tmpScore > 0.0]  = 1.0
            # tmpScore[(tmpScore < 0.0)] = 0.0
            # tmpScore[tradeDf[fet] > (tradeDf[fet].mean() - boundary)] = -1.0
            # tmpScore = np.sqrt(tmpScore)
            # print tmpScore
            # tradeDf['score'] += tmpScore * weightList[i]
            # tradeDf.loc[tradeDf[fet] <= boundary, 'score'] += 1.0

            # getTimeStats(tradeDf[tradeDf['score'] > 0], tradeDf)
            # tradeDf['score'] = 0.0

            raw_input('wait..')

        # boundary = getFetStats('score', tradeDf, plot = True)
        funcList = [len, accuracy, np.max, np.min, np.std, avgwin, avgloss, winratio, np.mean]
        statList[hold] = pivot_table(tradeDf, values = ['profit'], index = ['score'], aggfunc = funcList)
        statList[hold].columns = [s1 for (s1, s2) in statList[hold].columns.tolist()]

    # statDf = getMultiIndexDataFrame(fetStatList, levels = 3)
    # print statDf
    # statDf.to_clipboard()
    # # raw_input('copied to clipboard..')

    # statDf = getMultiIndexDataFrame(statList, levels = 2)
    # print statDf
    # statDf.to_clipboard()
    # raw_input('copied to clipboard..')