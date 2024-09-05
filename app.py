from flask import Flask, request, render_template
from datetime import datetime
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open('models/model.pkl', 'rb'))

# Allowed companies and date validation function
allowed_companies = ['AAPL', 'MSFT', 'GOOG', 'AMZN']

def validate_input(company, date):
    # Check if the company is valid
    if company not in allowed_companies:
        return False, f"Invalid company. Choose from {', '.join(allowed_companies)}."
    
    # Check if the date is valid (before or on the current date)
    try:
        input_date = datetime.strptime(date, '%Y-%m-%d').date()
        if input_date > datetime.now().date():
            return False, "Date cannot be in the future."
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD."
    
    return True, ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    date = request.form['date']
    
    # Validate inputs
    is_valid, message = validate_input(company, date)
    if not is_valid:
        return render_template('index.html', prediction_text=message)
    
    # Fetch stock data (dummy for now)
    input_data = [100.5, 102.0, 99.0, 101.5, 50000]  # Dummy data for example
    
    # Reshape and predict
    input_array = np.array(input_data).reshape(1, -1)  # Adjust for your model
    prediction = model.predict(input_array)
    
    return render_template('index.html', prediction_text=f'Predicted Stock Price for {company} on {date}: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
