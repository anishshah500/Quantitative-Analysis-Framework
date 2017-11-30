echo "Starting..."
# python execStrategy.py
python backtester.py
python visualise.py results/backtestLog.csv
python metrics.py results/backtestLog.csv > results/backtestMetrics.txt
echo "Done!"
