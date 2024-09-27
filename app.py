from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly
import json
from arima_optimizer import find_best_arima  # Import ARIMA optimizer
import random

app = Flask(__name__)

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None, "No data found for ticker."
    return data, None

# Function to generate random statistics for stock data
def generate_statistics(df):
    stats = {
        "Mean Close Price": round(df['Close'].mean(), 2),
        "Median Close Price": round(df['Close'].median(), 2),
        "Standard Deviation": round(df['Close'].std(), 2),
        "Highest Price": round(df['High'].max(), 2),
        "Lowest Price": round(df['Low'].min(), 2),
        "Total Trading Days": len(df),
    }
    return stats

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker'].upper()
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Get stock data
    df, error = get_stock_data(ticker, start_date, end_date)
    
    if error:
        return render_template('index.html', error=error)
    
    # Split data into train and test sets
    split_index = int(len(df['Close']) * 0.8)
    train = df['Close'].iloc[:split_index]
    test = df['Close'].iloc[split_index:]

    # Find ARIMA (p, d, q) values
    best_pdq = find_best_arima(ticker, start_date, end_date)

    # Fit ARIMA model
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=best_pdq)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    # Generate statistics
    statistics = generate_statistics(df)

    # Plot the data and forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Actual Prices', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    
    fig.update_layout(
        title=f'ARIMA Forecast vs. Actual Prices for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Data'
    )

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('index.html', pdq=best_pdq, graph_json=graph_json, statistics=statistics)

if __name__ == '__main__':
    app.run(debug=True)