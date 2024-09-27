import warnings
import itertools
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

# Function to find the best ARIMA (p, d, q) values
def find_best_arima(ticker, start_date, end_date):
    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    train = data['Close'].dropna()

    # Define range for p, d, q
    p = range(0, 4)
    d = range(0, 2)
    q = range(0, 4)
    
    pdq = list(itertools.product(p, d, q))
    best_aic = float("inf")
    best_order = None
    warnings.filterwarnings("ignore")
    
    for param in pdq:
        try:
            model = ARIMA(train, order=param)
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = param
        except:
            continue
    return best_order