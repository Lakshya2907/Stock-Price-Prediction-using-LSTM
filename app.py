from flask import Flask, request, render_template
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load your LSTM model and scaler
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = MinMaxScaler(feature_range=(0, 1))

allowed_companies = ['AAPL', 'MSFT', 'GOOG', 'AMZN','META','NFLX','TSLA','NVDA','INTC','CSCO']
from datetime import datetime, timedelta

def validate_input(date):
    try:
        input_date = datetime.strptime(date, '%Y-%m-%d').date()
        max_allowed_date = datetime.now().date() + timedelta(days=60)
        if input_date > max_allowed_date:
            return False, "Date cannot be more than 60 days in the future."
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD."
    
    return True, ""


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    prediction_date = request.form['date']
    
    is_valid, message = validate_input(prediction_date)
    if not is_valid:
        return render_template('index.html', prediction_text=message)
    
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download stock data for the last 60 days
    stock_symbol = company
    prediction_date_obj = datetime.strptime(prediction_date, '%Y-%m-%d')
    start_date = (prediction_date_obj - timedelta(days=80)).strftime('%Y-%m-%d')
    historical_data = yf.download(stock_symbol, start=start_date, end=prediction_date_obj)
    if historical_data.empty:
        return render_template('index.html', prediction_text=f"No data found for {company}.")

    historical_data = historical_data.sort_index()
    # Prepare the data for prediction (scale and reshape)
    scaled_data = scaler.fit_transform(historical_data[['Close']])
    sequence_length = 60
    last_sequence = scaled_data[-sequence_length:]
    x_input = np.array([last_sequence])
    x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))  # Reshape for LSTM
    
    # Make prediction using LSTM model
    prediction = model.predict(x_input)
    predicted_price = scaler.inverse_transform(prediction)
    
    return render_template('index.html', prediction_text=f'Predicted Stock Price for {company} on {prediction_date}: {predicted_price[0][0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
