import sys
import numpy as np
import scipy
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
from metrics import *

def print_NAV_graph(file, statsDict = None, navId = ''):
    #opening the file and reading the content and storing it in a pandas data frame

    df = pd.read_csv(file)
    df['time'] = df['time'].map(lambda t: time.strftime("%a, %d %b %Y", time.localtime(t)))

    xLab = list(df['time'])
    x = range(len(xLab))
    val = list(df['value'] - df['value'].iloc[0])
    budget = list(df['currBudget'] - df['currBudget'].iloc[0])
    pnl = list(df['currPnl'] - df['currPnl'].iloc[0])
    tradeCost = list(df['tradeCost'].cumsum())
    pnlNoCost = list((df['value'] - df['value'].iloc[0]) + df['tradeCost'].cumsum())
    
    period = (len(x)/10)

    # plt.figure(figsize=(100,100))
    fig, ax = plt.subplots(1,1, figsize = (10,7)) 
    ax.grid(True)

    # ax.plot(x, val, 'r')
    ax.plot(x, val, label = 'Value', color = 'g', alpha = 0.8)
    ax.plot(x, pnl, label = 'CurrPnl', color = 'b', alpha = 0.8)
    ax.plot(x, pnlNoCost, label = 'Value w/o Cost', color = 'c', alpha = 0.8)
    ax.plot(x, tradeCost, label = 'Trade Cost', color = 'y', alpha = 0.8)
    ax.set_xticks(x[::period])
    ax.set_xticklabels(xLab[::period], rotation=40, fontsize=7)
    ax.set_xlabel('Date-Time')
    ax.set_ylabel('Value')
    ax.set_ylim([-15000, 20000])
    ax.set_title('NAV Graph: ' + navId)
    ax.legend(loc='lower right')

    if (statsDict is not None):
        statsText = ''
        statsText += ("%-18s: %12.2f" % ('Total PnL', statsDict['PnL'])) + '\n'
        statsText += ("%-16s: %12.3f" % ('Annual Ret', statsDict['annualRet'])) + '\n'
        statsText += ("%-11s: %12.3f" % ('maxDrawdown', statsDict['maxDrawdown'])) + '\n'
        statsText += ("%-18s: %12.3f" % ('StdDev', statsDict['AnnualizedStdDev'])) + '\n'
        statsText += ("%-15s: %12.2f" % ('Calmar Ratio', statsDict['CalmarRatio'])) + '\n'
        statsText += ("%-15s: %12.2f" % ('Sharpe Ratio', statsDict['Sharpe']))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, statsText, transform = ax.transAxes, fontsize = 8, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig("results/gapNAV/gapsPlot_v2/" + navId + ".svg")
    # plt.show()

def makeNAVList(logList, labelList, statsDict = None, comments = False, navId = ''):
    #opening the file and reading the content and storing it in a pandas data frame

    fig, ax = plt.subplots(1, 1, figsize = (10,7)) 
    colorList = ['r', 'b', 'y', 'g']

    for i in range(len(logList)):

        fileName = logList[i]

        df = pd.read_csv(fileName)
        df['time'] = df['time'].map(lambda t: time.strftime("%a, %d %b %Y", time.localtime(t)))

        xLab = list(df['time'])
        x = range(len(xLab))
        # val = list(df['value'] - df['value'].iloc[0])
        # budget = list(df['currBudget'] - df['currBudget'].iloc[0])
        # pnl = list(df['currPnl'] - df['currPnl'].iloc[0])
        # pnlNoCost = list((df['value'] - df['value'].iloc[0]) + df['tradeCost'].cumsum())
        val = list(df['value'])
        budget = list(df['currBudget'])
        pnl = list(df['currPnl'])
        pnlNoCost = list(df['value'] + df['tradeCost'].cumsum())
        
        period = (len(x)/10)
        ax.plot(x, pnlNoCost, label = labelList[i], color = colorList[i], alpha = 0.8)

        if (statsDict is not None):
            statsText = 'Stats for: ' + labelList[i] + '\n'
            statsText += ("%-18s: %12.2f" % ('Total PnL', statsDict[i]['PnL'])) + '\n'
            statsText += ("%-16s: %12.4f" % ('Annual Ret', statsDict[i]['annualRet'])) + '\n'
            statsText += ("%-11s: %12.4f" % ('maxDrawdown', statsDict[i]['maxDrawdown'])) + '\n'
            statsText += ("%-18s: %12.4f" % ('StdDev', statsDict[i]['AnnualizedStdDev'])) + '\n'
            statsText += ("%-15s: %12.2f" % ('Calmar Ratio', statsDict[i]['CalmarRatio'])) + '\n'
            statsText += ("%-15s: %12.2f" % ('Sharpe Ratio', statsDict[i]['Sharpe']))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.82 - (i*0.17), statsText, transform = ax.transAxes, fontsize = 8, verticalalignment='top', bbox=props)

    ax.grid(True)
    ax.set_xticks(x[::period])
    ax.set_xticklabels(xLab[::period], rotation=40, fontsize=7)
    ax.set_xlabel('Date-Time')
    ax.set_ylabel('Value')
    ax.set_title('NAV Graph: ' + navId)
    ax.legend(loc='lower right')

    if (comments):
        statsText = 'Note that the number of trades \nperformed by each strategy is different.\n'
        statsText += 'Therefore the respective pnls based\n on only the 0-cost should not be taken\n on face value.'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, statsText, transform = ax.transAxes, fontsize = 8, verticalalignment='top', bbox=props)

    plt.tight_layout()
    # plt.savefig("results/gapNAV/gapsPlot_v2/" + navId + ".svg")
    plt.show()

if __name__ == '__main__':
    # print_NAV_graph(sys.argv[1])
    filePath = 'results/gapBacktestLog/'    
    labelList = ['AllFilters', 'LowHighRange', 'GapRatio', 'VolDailyRank']
    fileList = [filePath + 'backtestLog' + x + '.csv' for x in labelList]

    config = {}
    config['NUM_DAYS'] = 1458
    config['LOG_PERIOD'] = 9000
    statsDict = []
    for fileName in fileList:
        tmpStats = get_standard_ratios(fileName, config)
        statsDict.append(tmpStats)

    makeNAVList(fileList, labelList, statsDict = statsDict, comments = True, navId = 'Only w/o Costs')
